"""Current-state UAV depth replanning with actual cascaded-controller physics."""
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from scipy.spatial.transform import Rotation
from uav_ac.control.trajectory_controller import TrajectoryReference
from uav_ac.yopo.uav.deployment import NED_TO_NWU
from uav_ac.yopo.vendor.yopo_minco.policy.poly_solver import calculate_yaw


class UavNavigation:
    def __init__(self,simulation,controller,planner,observer,settings,control_dt,duration,*,visualize=False,executor=None):
        self.sim=simulation;self.quad=simulation.quad;self.controller=controller;self.planner=planner;self.observer=observer
        self.settings=settings;self.control_dt=control_dt;self.duration=duration;self.visualize=visualize
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='yopo-uav') if executor is None else executor
        self.stride=int(round(control_dt/self.quad.dt));self.index=0;self.control_index=0
        self.state='WAITING_PLAN';self.failure_reason=None;self.accepted=0;self.requests=0
        self.plan=None;self.plan_start=0.;self.future=None;self.capture_time=0.;self.submit_wall=0.;self.next_plan=0.
        self.goal=np.array(self.sim.goal_position,copy=True);self.row=self.hold(self.quad.position)
        rotation=Rotation.from_quat(self.quad.X[[4,5,6,3]]).as_matrix()
        self.last_yaw=-float(np.arctan2(rotation[1,0],rotation[0,0]))
        self.previous_velocity=self.quad.X[7:10].copy();self.previous_time=self.sim.time;self.acceleration=np.zeros(3)
        self.stable_since=None;self.brake_hold=None;self.last_log=-np.inf
        self.max_tracking_error=0.;self.max_speed=0.;self.terminal_state=None;self.stopped_at=None

    @property
    def done(self):return self.state in ('FAILED','REACHED')

    def hold(self,position):return np.r_[np.array(position,copy=True),np.zeros(6),0.]

    def finish(self,state,reason=None):
        if self.done:return
        self.state=state;self.failure_reason=reason;self.stopped_at=self.sim.time;self.terminal_state=self.quad.X.copy()
        self.row=self.hold(self.quad.position);self.row[9]=-self.last_yaw
        if self.future is not None:self.future.cancel()
        print(f'UAV YOPO terminal: t={self.sim.time:.2f}s state={state} reason={reason}',flush=True)

    def close(self):
        if self.future is not None:self.future.cancel()
        self.executor.shutdown(wait=False,cancel_futures=True)

    def _poll(self,now):
        if self.future is None:return
        wall_age=time.perf_counter()-self.submit_wall
        if not self.future.done():
            if wall_age>self.settings.planning_timeout:self.finish('FAILED','planning_timeout')
            return
        try:
            plan=self.future.result();age=now-self.capture_time
            if wall_age>self.settings.planning_timeout:raise TimeoutError('planning_timeout')
            if age>self.settings.max_plan_age or age>=plan.trajectory.total_time:raise TimeoutError('stale_plan')
            self.plan=plan;self.plan_start=self.capture_time;self.accepted+=1;self.state='TRACKING'
            if self.visualize:
                paths=[];colors=[]
                t=np.linspace(0,plan.trajectory.total_time,40)
                paths.append(plan.trajectory.position(t)@NED_TO_NWU);colors.append([.1,1.,.3,1.])
                if hasattr(plan,'trajectories'):
                    choices=np.argsort(plan.scores)[-6:][::-1]
                    all_paths=plan.trajectories.position(t)@NED_TO_NWU
                    for i in choices:
                        if i==plan.selected_index:continue
                        paths.append(all_paths[i]);colors.append([.75,.8,.85,.45])
                        if len(paths)==6:break
                self.sim.set_planning_paths(paths,colors)
        except Exception as error:
            self.finish('FAILED',str(error) if isinstance(error,TimeoutError) else f'inference_error:{error}')
        finally:self.future=None

    def _tick(self,now):
        if self.sim.collision_detected:self.finish('FAILED','simulation_collision');return
        if now>=self.duration:self.finish('FAILED','episode_timeout');return
        state=self.quad.X.copy()
        if not np.isfinite(state).all():self.finish('FAILED','nonfinite_state');return
        bounds=self.sim.space_limits
        if bounds is not None and (np.any(state[:2]<bounds[0,:2]-.02) or np.any(state[:2]>bounds[1,:2]+.02) or state[2]>bounds[1,2]+.02):
            self.finish('FAILED','actual_workspace_violation');return
        dt=now-self.previous_time
        if dt>0:self.acceleration=.8*self.acceleration+.2*(state[7:10]-self.previous_velocity)/dt
        self.previous_velocity=state[7:10].copy();self.previous_time=now
        self.max_speed=max(self.max_speed,float(np.linalg.norm(state[7:10])))
        error=float(np.linalg.norm(state[:3]-self.goal))
        if error<=self.settings.goal_tolerance:
            if self.state!='SETTLING':
                if self.future is not None:self.future.cancel()
                self.future=None;self.plan=None;self.next_plan=now
                self.row=self.hold(state[:3]);self.row[9]=-self.last_yaw
            self.state='SETTLING'
            if np.linalg.norm(state[7:10])<=self.settings.velocity_tolerance:
                if self.stable_since is None:self.stable_since=now
                if now-self.stable_since>=self.settings.settle_time:self.finish('REACHED');return
            else:self.stable_since=None
            return
        self.stable_since=None
        if self.state=='SETTLING':self.state='TRACKING'
        self._poll(now)
        if self.done:return
        if self.future is None and now>=self.next_plan:
            depth,_=self.observer.observe();snapshot=self.quad.X.copy()
            self.capture_time=self.sim.time;self.submit_wall=time.perf_counter()
            self.future=self.executor.submit(self.planner.plan,depth,snapshot,self.acceleration.copy(),self.goal.copy())
            self.requests+=1;self.next_plan=now+self.settings.replan_dt
            self._poll(now)
        if self.plan is not None:
            age=now-self.plan_start
            if age>=self.plan.trajectory.total_time:
                self.finish('FAILED','trajectory_exhausted');return
            if self.plan.brake:
                if self.brake_hold is None:self.brake_hold=self.row[:3].copy()
                goal_dir=NED_TO_NWU@(self.goal-self.brake_hold)
                yaw,_=calculate_yaw(goal_dir,goal_dir,self.last_yaw,self.control_dt)
                self.row=self.hold(self.brake_hold)
            else:
                self.brake_hold=None;t=self.plan.trajectory
                p,v,a=t.position(age),t.velocity(age),t.acceleration(age)
                yaw,_=calculate_yaw(v,NED_TO_NWU@self.goal-p,self.last_yaw,self.control_dt)
                self.row=np.r_[NED_TO_NWU@p,NED_TO_NWU@v,NED_TO_NWU@a,0.]
            self.last_yaw=float(yaw);self.row[9]=-yaw
        tracking=float(np.linalg.norm(state[:3]-self.row[:3]));self.max_tracking_error=max(self.max_tracking_error,tracking)
        if tracking>2.:self.finish('FAILED','tracking_diverged');return
        if now-self.last_log>=self.settings.log_dt:
            print(f'UAV YOPO: t={now:.2f}s state={self.state} replans={self.accepted} goal_error={error:.2f}m tracking={tracking:.3f}m',flush=True);self.last_log=now

    def step(self):
        tick=self.index%self.stride==0
        if tick:
            if not self.done:self._tick(self.sim.time)
            self.control_index+=1
        rows=np.broadcast_to(self.row,(self.control_index+1,10))
        reference=TrajectoryReference(rows,self.control_index,self.control_dt,is_control_tick=tick)
        command=self.controller.step(self.quad,reference)
        self.quad.set_propeller_speed(command.thrust,command.moment);self.index+=1

    def result(self):
        state=self.quad.X if self.terminal_state is None else self.terminal_state
        return dict(state=self.state,success=self.state=='REACHED',failure_reason=self.failure_reason,
            simulation_time=self.sim.time if self.stopped_at is None else self.stopped_at,
            goal_error=float(np.linalg.norm(state[:3]-self.goal)),position_ned=state[:3].tolist(),
            collision=bool(self.sim.collision_detected),accepted_plans=self.accepted,replan_requests=self.requests,
            maximum_base_tracking_error_m=self.max_tracking_error,maximum_actual_speed_mps=self.max_speed)
