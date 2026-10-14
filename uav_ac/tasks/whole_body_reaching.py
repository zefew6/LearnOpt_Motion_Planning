"""Online reaching: actual physics state → single-frame plan → reference tracking."""
from concurrent.futures import Future
from queue import Queue
from threading import Thread
import time
import numpy as np
from scipy.spatial.transform import Rotation
from uav_ac.yopo.contracts import PlannerObservation
from uav_ac.yopo.model.encoding import wrap
from uav_ac.planning.trajectory.aerial_manipulator_minco.flatness import recover_state
from uav_ac.robot.aerial_manipulator import AerialManipulatorReference


class PlannerWorker:
    """One daemon worker: an overdue inference cannot block viewer shutdown."""
    def __init__(self):
        self.queue=Queue();self.closed=False
        self.thread=Thread(target=self._run,name='whole-body-plan',daemon=True)
        self.thread.start()
    def _run(self):
        while True:
            job=self.queue.get()
            if job is None:return
            future,fn,args=job
            if not future.set_running_or_notify_cancel():continue
            try:future.set_result(fn(*args))
            except BaseException as error:future.set_exception(error)
    def submit(self,fn,*args):
        if self.closed:raise RuntimeError('planner worker is closed')
        future=Future();self.queue.put((future,fn,args));return future
    def shutdown(self,*,wait=False,cancel_futures=True):
        self.closed=True
        if cancel_futures:
            while not self.queue.empty():
                job=self.queue.get_nowait()
                if job is not None:job[0].cancel()
        self.queue.put(None)
        if wait:self.thread.join()


def heading_and_rate(configuration,velocity):
    q=configuration[3:7]
    rotation=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    y=rotation[:,1];dy=rotation@np.array([-velocity[5],0.,velocity[3]])
    norm=y[0]**2+y[1]**2
    if norm<1e-8:raise ValueError('body heading is singular')
    return float(np.arctan2(-y[0],y[1])),float((y[0]*dy[1]-y[1]*dy[0])/norm)


class ActualStateSampler:
    """Finite-difference derivatives from measured state, never the planned state."""
    def __init__(self):self.previous=None;self.acceleration=np.zeros(8);self.jerk=np.zeros(8)
    def update(self,robot,now):
        q=robot.configuration.copy();v=robot.velocity.copy();yaw,rate=heading_and_rate(q,v)
        position=np.r_[q[:3],yaw,q[7:11]];speed=np.r_[v[:3],rate,v[6:10]]
        if self.previous is not None:
            dt=now-self.previous[0]
            if dt<=0:raise ValueError('actual-state timestamps must increase')
            old=self.acceleration.copy();alpha=dt/(.10+dt)
            self.acceleration=(1-alpha)*old+alpha*(speed-self.previous[1])/dt
            self.jerk=(self.acceleration-old)/dt
        self.previous=(float(now),speed.copy())
        return q,v,np.vstack((position,speed,self.acceleration)),self.jerk.copy()


class WholeBodyReaching:
    def __init__(self,simulation,controller,planner,observer,goal,gap,settings,control_dt,duration,
                 *,on_plan=None,executor=None):
        self.simulation=simulation;self.robot=simulation.robot;self.controller=controller
        self.planner=planner;self.observer=observer;self.goal=np.asarray(goal,dtype=float).copy();self.gap=gap
        self.settings=settings;self.control_dt=control_dt;self.duration=duration;self.on_plan=on_plan
        self.stride=int(round(control_dt/simulation.quad.dt));self.index=0
        self.executor=PlannerWorker() if executor is None else executor
        self.sampler=ActualStateSampler();self.future=None;self.capture_time=0.;self.submit_wall=0.
        self.plan=None;self.plan_start=0.;self.next_plan=0.;self.stable_since=None
        self.state='WAITING_PLAN';self.failure_reason=None;self.stopped_at=None;self.terminal=None
        self.terminal_collision=None;self.terminal_saturations=None
        self.request_count=0;self.accepted_count=0;self.latencies=[];self.plan_ages=[];self.depth_latencies=[]
        self.max_base_speed=0.;self.max_base_error=0.;self.max_joint_error=0.;self.max_rate=0.;self.max_reference_rate_jump=0.
        self.positions=[];self.last_progress_time=simulation.time;self.progress_position=self.robot.configuration[:3].copy()
        self.next_log=0.;self.command_reference=self._hold(self.robot.configuration)

    @property
    def done(self):return self.state in ('REACHED','FAILED')

    def _hold(self,q):
        return AerialManipulatorReference(np.asarray(q).copy(),np.zeros(11),np.zeros(11))

    def _finish(self,state,reason=None):
        if self.done:return
        self.state=state;self.failure_reason=reason;self.stopped_at=self.simulation.time
        print(f'Whole-body terminal: t={self.stopped_at:.2f}s state={state} reason={reason}',flush=True)
        self.terminal=(self.robot.configuration.copy(),self.robot.velocity.copy())
        self.terminal_collision=self.simulation.collision_detected
        self.terminal_saturations=self.controller.saturation_count
        self.command_reference=self._hold(self.robot.configuration)
        if self.future is not None:self.future.cancel()

    def _errors(self,q,v):
        yaw,_=heading_and_rate(q,v)
        return {'position':float(np.linalg.norm(q[:3]-self.goal[:3])),
                'yaw':abs(float(wrap(yaw-self.goal[3]))),'joints':float(np.max(np.abs(q[7:11]-self.goal[4:])))}

    def abort(self,reason):self._finish('FAILED',reason)

    def _poll(self,now):
        if self.future is None:return
        wall_age=time.perf_counter()-self.submit_wall
        if not self.future.done():
            if wall_age>self.settings.planning_timeout:self._finish('FAILED','planning_timeout')
            return
        try:
            result=self.future.result();trajectory=result.trajectory
            age=now-self.capture_time
            if wall_age>self.settings.planning_timeout:raise TimeoutError('planning_timeout')
            if age>self.settings.max_plan_age or age>=trajectory.total_time:raise TimeoutError('stale_plan')
            reference=trajectory.reference(age,self.robot,gripper_opening=self.gap)
            self.max_reference_rate_jump=max(self.max_reference_rate_jump,float(np.linalg.norm(reference.velocity[3:6]-self.robot.velocity[3:6])))
            self.plan=trajectory;self.plan_start=self.capture_time;self.accepted_count+=1
            self.latencies.append(wall_age);self.plan_ages.append(age);self.state='TRACKING'
            if self.on_plan is not None:self.on_plan(result,age)
        except (ValueError,RuntimeError,FloatingPointError,TimeoutError,np.linalg.LinAlgError) as error:
            self._finish('FAILED',f'planning_error:{error}')
        finally:self.future=None

    def step(self):
        now=self.simulation.time
        if not self.done and self.index%self.stride==0:
            try:self._tick(now)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                self._finish('FAILED',f'control_or_observation_error:{error}')
        self.robot.apply(self.controller.step(self.command_reference))
        self.index+=1

    def _tick(self,now):
        if self.simulation.collision_detected:
            self._finish('FAILED','simulation_collision');return
        if now>=self.duration:
            self._finish('FAILED','episode_timeout');return
        q,v,pva,jerk=self.sampler.update(self.robot,now)
        if not np.isfinite(q).all() or not np.isfinite(v).all():self._finish('FAILED','nonfinite_actual_state');return
        bounds=self.simulation.space_limits
        if np.any(q[:3]<bounds[0]-.02) or np.any(q[:3]>bounds[1]+.02):
            self._finish('FAILED','actual_workspace_violation');return
        self.max_base_speed=max(self.max_base_speed,float(np.linalg.norm(v[:3])))
        errors=self._errors(q,v);s=self.settings
        near=errors['position']<=s.position_tolerance and errors['yaw']<=s.yaw_tolerance and errors['joints']<=s.joint_tolerance
        if self.state=='SETTLING' or near:
            self.state='SETTLING'
            target,_=recover_state(np.vstack((self.goal,np.zeros((3,8)))),self.gap)
            self.command_reference=self._hold(target)
            if self.future is not None:self.future.cancel();self.future=None
            stable=near and np.linalg.norm(v[:3])<=s.velocity_tolerance and np.linalg.norm(v[3:6])<=s.body_rate_tolerance and np.max(abs(v[6:10]))<=s.joint_rate_tolerance
            self.stable_since=now if stable and self.stable_since is None else self.stable_since if stable else None
            if self.stable_since is not None and now-self.stable_since>=s.settle_time:self._finish('REACHED')
        else:
            self._poll(now)
            if self.done:return
            if self.future is None and now>=self.next_plan:
                started=time.perf_counter();depth,mask=self.observer.observe();self.depth_latencies.append(time.perf_counter()-started)
                observation=PlannerObservation(depth,mask,q,v,pva,self.goal,self.observer.spec,now,jerk,'simulation')
                self.capture_time=now;self.submit_wall=time.perf_counter()
                self.future=self.executor.submit(self.planner.plan,observation)
                self.request_count+=1;self.next_plan=now+s.replan_dt
            if self.plan is not None:
                local=now-self.plan_start
                if local>=self.plan.total_time:self._finish('FAILED','reference_horizon_exhausted');return
                self.command_reference=self.plan.reference(local,self.robot,gripper_opening=self.gap)
            if np.linalg.norm(q[:3]-self.progress_position)>=s.progress_distance:
                self.progress_position=q[:3].copy();self.last_progress_time=now
            elif now-self.last_progress_time>=s.stall_time:self._finish('FAILED','no_motion_progress');return
        base_error=float(np.linalg.norm(q[:3]-self.command_reference.configuration[:3]))
        self.max_base_error=max(self.max_base_error,base_error)
        self.max_joint_error=max(self.max_joint_error,float(np.max(abs(q[7:11]-self.command_reference.configuration[7:11]))))
        self.max_rate=max(self.max_rate,float(np.max(abs(v[6:10]))));self.positions.append(q[:3].copy())
        if base_error>s.tracking_error_limit:self._finish('FAILED','tracking_diverged');return
        if now>=self.next_log:
            print(f'Whole-body online: t={now:.2f}s state={self.state} replans={self.accepted_count} goal_error={errors["position"]:.2f}m tracking={base_error:.3f}m speed={np.linalg.norm(v[:3]):.2f}m/s',flush=True)
            self.next_log=now+s.log_dt

    def result(self):
        q,v=self.terminal if self.terminal is not None else (self.robot.configuration,self.robot.velocity)
        return {'state':self.state,'success':self.state=='REACHED','failure_reason':self.failure_reason,
                'collision':self.simulation.collision_detected if self.terminal_collision is None else self.terminal_collision,'simulation_time':self.stopped_at if self.stopped_at is not None else self.simulation.time,
                'replan_requests':self.request_count,'accepted_plans':self.accepted_count,'goal_errors':self._errors(q,v),
                'maximum_actual_base_speed_mps':self.max_base_speed,'maximum_base_tracking_error_m':self.max_base_error,'maximum_joint_tracking_error_rad':self.max_joint_error,
                'maximum_actual_joint_rate':self.max_rate,'maximum_handoff_body_rate_difference':self.max_reference_rate_jump,
                'mean_planning_wall_seconds':float(np.mean(self.latencies)) if self.latencies else None,
                'max_plan_age_sim_seconds':max(self.plan_ages,default=0.),'controller_saturation_steps':self.controller.saturation_count if self.terminal_saturations is None else self.terminal_saturations,
                'actual_path_ned':np.asarray(self.positions).reshape(-1,3).tolist(),
                'interpretation':'actual MuJoCo physics + whole-body controller; online single-frame replanning, no map-based candidate filtering'}

    def close(self):
        if self.future is not None:self.future.cancel()
        self.executor.shutdown(wait=False,cancel_futures=True)
