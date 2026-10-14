"""YOPO-MINCO focused guidance, extended to actual whole-body geometry.

Pinned reference: d5e1e60f83cc991602f142d9b6b96204a24d6b9f.
Collision detection is discrete whole-body sampling, not a free-ball certificate.
Detached goal/curvature derivatives are recorded separately for honest FD audits.
"""
from dataclasses import dataclass, fields
import numpy as np
from scipy.special import logsumexp
from uav_ac.yopo.training.legacy import DirectEvaluator, GuidanceBatch
from uav_ac.yopo.trajectory.minco import world_problem
from uav_ac.planning.trajectory.gcopter.optimization import evaluate_minco_objective
from uav_ac.planning.trajectory.gcopter.mappings import polynomial_basis_matrix
from uav_ac.planning.trajectory.aerial_manipulator_minco.constraints import _flatness_body_rate_squared

@dataclass(frozen=True)
class MincoGuidanceConfig:
    safety_weight: float = 1.
    self_weight: float = 1.
    smoothness_weight: float = .002
    acceleration_weight: float = .002
    feasibility_weight: float = 1.5
    time_weight: float = 1.5
    goal_weight: float = .6
    goal_perpendicular_weight: float = .3
    goal_pose_weight: float = 1.
    safety_beta: float = 4.
    environment_safe: float = .15
    environment_d0: float = .05
    environment_decay: float = .05
    self_safe: float = .015
    self_decay: float = .015
    curvature_weight: float = 1.
    lateral_acceleration_ratio: float = .8
    jerk_max: float = 30.
    samples_per_piece: int = 17

    def __post_init__(self):
        positive={'safety_beta','environment_safe','environment_decay','self_decay','jerk_max','lateral_acceleration_ratio'}
        for f in fields(self):
            value=getattr(self,f.name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0 or (f.name in positive and value==0):
                raise ValueError(f'invalid guidance {f.name}')
        if not isinstance(self.samples_per_piece,int) or self.samples_per_piece<3:raise ValueError('guidance samples_per_piece must be >=3')
        if not 0<self.lateral_acceleration_ratio<1:raise ValueError('lateral_acceleration_ratio must lie in (0,1)')

    @classmethod
    def from_mapping(cls,values):
        if not isinstance(values,dict) or set(values)-{f.name for f in fields(cls)}:raise ValueError('unknown guidance settings')
        return cls(**values)


def _huber(x):
    x=np.asarray(x);return float(np.where(abs(x)<1,.5*x*x,abs(x)-.5).sum()),np.clip(x,-1,1)


def goal_cost_gradient(head,tail,goal,settings,*,collided=False):
    delta=goal[:3]-head[:3];direction=delta/max(np.linalg.norm(delta),1e-6)
    error=tail[:3]-goal[:3];parallel=np.dot(error,direction)*direction;perpendicular=error-parallel
    cp,gp=_huber(parallel);cl,gl=_huber(perpendicular);gradient=np.zeros(8)
    gradient[:3]=np.dot(gp,direction)*direction+settings.goal_perpendicular_weight*(gl-np.dot(gl,direction)*direction)
    yaw=np.arctan2(np.sin(tail[3]-goal[3]),np.cos(tail[3]-goal[3]));cy,gy=_huber(yaw/.7)
    cq,gq=_huber((tail[4:]-goal[4:])/.8)
    gradient[3]=settings.goal_pose_weight*gy/.7;gradient[4:]=settings.goal_pose_weight*gq/.8
    value=settings.goal_weight*(cp+settings.goal_perpendicular_weight*cl+settings.goal_pose_weight*(cy+cq))
    return value,np.zeros(8) if collided else settings.goal_weight*gradient


def focused_safety(cost,contacts,*,beta):
    cost=np.asarray(cost);contacts=np.asarray(contacts,dtype=bool);weight=np.zeros_like(cost)
    collided=bool(contacts.any())
    if collided:
        index=max(0,int(np.flatnonzero(contacts)[0])-1);weight[index]=1.;return float(cost[index]),weight,True
    logits=beta*cost;normalizer=logsumexp(logits);weight=np.exp(logits-normalizer)
    return float((normalizer-np.log(len(cost)))/beta),weight,False


def curvature_cost_gradient(v,a,vmax,amax,ratio):
    """Return value, speed-only backward, and intentionally detached cap derivative."""
    speed=np.linalg.norm(v);cross=np.cross(v,a);norm=np.linalg.norm(cross);safe=max(speed,.5)
    curvature=norm/safe**3;cap0=np.sqrt(ratio*amax/max(curvature,1e-6));cap=np.clip(cap0,.5,vmax)
    gradient=np.zeros((2,3));detached=np.zeros((2,3));value=max(0.,speed-cap)/vmax
    if value==0:return value,gradient,detached
    gradient[0]=v/max(speed,1e-8)/vmax
    if .5<cap0<vmax and curvature>1e-6 and norm>1e-8:
        normal=cross/norm
        dk_v=np.cross(a,normal)/safe**3
        if speed>.5:dk_v-=3*norm/safe**4*v/speed
        dk_a=np.cross(normal,v)/safe**3
        detached[0]=.5*cap0/curvature*dk_v/vmax;detached[1]=.5*cap0/curvature*dk_a/vmax
    return value,gradient,detached


def acceleration_energy(coefficients,times,weights):
    """Exact polynomial integral and fixed-coefficient duration partials."""
    multiplier=np.array([2.,6.,12.,20.]);power=np.arange(4);cost=0.;gc=np.zeros_like(coefficients);gt=np.zeros_like(times)
    for piece,(coeff,time) in enumerate(zip(coefficients,times)):
        acceleration=multiplier[:,None]*coeff[2:]
        exponent=power[:,None]+power[None,:]+1
        gram=time**exponent/exponent
        cost+=np.sum(weights*acceleration*(gram@acceleration))
        gc[piece,2:]=2*multiplier[:,None]*(gram@acceleration)*weights
        gt[piece]=np.sum(weights*(time**power@acceleration)**2)
    return float(cost),gc,gt


class MincoGuidanceEvaluator(DirectEvaluator):
    @classmethod
    def from_simulation(cls,simulation,*,gripper_gap=.06,config=None,guidance=None):
        result=super().from_simulation(simulation,gripper_gap=gripper_gap,config=config,collision_model='exact_geometry')
        result.guidance=MincoGuidanceConfig() if guidance is None else guidance
        return result

    def _clearance_samples(self,sigma,acceleration):
        e=self.evaluator;settings=self.guidance
        positions,_,_,attitude,_,prepared=e._full_pose_geometry(sigma,acceleration)
        # Complete geometry sphere covers only prune far world pairs. No binary
        # voxel hit is used as a contact or as the differentiable safety cost.
        closest=np.minimum(np.maximum(positions[:,None,:],e.world_aabb_lower),e.world_aabb_upper)
        near=np.sum((positions[:,None,:]-closest)**2,axis=2)<=(e.sphere_radii[:,None]+settings.environment_safe)**2+1e-12
        sphere,world=np.nonzero(near & (e.sphere_geom_indices[:,None]>=0))
        world_ids=e._world_query_table[e.sphere_geom_indices[sphere],world];world_ids=world_ids[world_ids>=0]
        indices=np.unique(np.r_[world_ids,e._self_query_indices])
        exact=self.robot.exact_collision_distances(None,pairs=e._query_pairs,pair_indices=indices,prepared=prepared,
            with_jacobians=True,with_pose_jacobians=True,check_limits=False,
            jacobian_distance_thresholds={e._query_pairs.keys[i]:settings.environment_safe if e._query_base_kinds[i]==1 else settings.self_safe for i in indices})
        kinds=e._query_base_kinds[indices];distance=exact['distances'];pose=exact['jacobians']
        values=[];gradients=[];contacts=[]
        for kind,margin,origin,decay,weight in ((1,settings.environment_safe,settings.environment_d0,settings.environment_decay,settings.safety_weight),
                                               (2,settings.self_safe,0.,settings.self_decay,settings.self_weight)):
            ids=np.flatnonzero(kinds==kind);gradient=np.zeros((4,8));cost=0.;contact=False
            if len(ids):
                index=ids[np.argmin(distance[ids])];d=float(distance[index]);contact=d<=0.
                if d<margin:
                    # Only last-safe samples contribute to a colliding trajectory;
                    # cap the deep-initial-contact exponent to keep bad inputs finite.
                    exponent=np.exp(np.clip((origin-d)/decay,-700,50))
                    cost=weight*(exponent-np.exp((origin-margin)/decay))
                    factor=-weight*exponent/decay if (origin-d)/decay<50 else 0.
                    j=pose[index];gradient[0]=factor*np.r_[j[:3],j[3:6]@attitude[:,3],j[6:10]]
                    gradient[2,:3]=factor*(j[3:6]@attitude[:,:3])
            values.append(float(cost));gradients.append(gradient);contacts.append(contact)
        return values,gradients,contacts

    def _physical_residuals(self,values):
        s,v,a,j=values[:4];cfg=self.config;quad=self.evaluator.quad;rows=[];gradients=[]
        def add(value,order,gradient):
            g=np.zeros((4,8));g[order]=gradient;rows.append(float(value));gradients.append(g)
        eye=np.eye(8)
        for vector,limit,order in ((v[:3],cfg.max_speed,1),(a[:3],cfg.max_acceleration,2),(j[:3],self.guidance.jerk_max,3)):
            norm=np.linalg.norm(vector);add(norm/limit-1,order,np.r_[vector/max(norm,1e-8)/limit,np.zeros(5)])
        for value,limit,order in ((v[3],cfg.max_yaw_rate,1),(a[3],cfg.max_yaw_acceleration,2)):
            add(abs(value)/limit-1,order,eye[3]*np.sign(value)/limit)
        for i in range(4):
            k=i+4
            add((s[k]-self.robot.limits.joint_upper[i])/.8,0,eye[k]/.8)
            add((self.robot.limits.joint_lower[i]-s[k])/.8,0,-eye[k]/.8)
            for vector,limits,order in ((v,cfg.joint_velocity_limits,1),(a,cfg.joint_acceleration_limits,2)):
                add(abs(vector[k])/limits[i]-1,order,eye[k]*np.sign(vector[k])/limits[i])
        force=np.array([-a[0],-a[1],quad.g-a[2]]);magnitude=max(np.linalg.norm(force),1e-8);axis=force/magnitude
        thrust=self.evaluator.mass*magnitude;maximum=4*quad.max_thrust;gt=np.r_[-self.evaluator.mass*axis,np.zeros(5)]/maximum
        add((thrust-maximum)/maximum,2,gt);add((4*quad.min_thrust-thrust)/maximum,2,-gt)
        cosine=np.clip(axis[2],-1,1);sine=np.sqrt(max(0.,1-cosine*cosine));tilt=np.arccos(cosine)
        ga=(np.array([0.,0.,1.])-cosine*axis)/(magnitude*max(sine,1e-8)*quad.max_tilt_angle)
        add(tilt/quad.max_tilt_angle-1,2,np.r_[ga,np.zeros(5)])
        rate2,dr=_flatness_body_rate_squared(a[:3],j[:3],s[3],v[3],quad.g)
        g=np.zeros((4,8));g[0,3]=dr[6];g[1,3]=dr[7];g[2,:3]=dr[:3];g[3,:3]=dr[3:6]
        rows.append(rate2/cfg.max_body_rate**2-1);gradients.append(g/cfg.max_body_rate**2)
        for sign,bound in ((1,self.evaluator.bounds[1]),(-1,self.evaluator.bounds[0])):
            for i in range(3):add(sign*(s[i]-bound[i])/.5,0,sign*eye[i]/.5)
        return np.asarray(rows),np.asarray(gradients)

    def _sample_term(self,times,coeff):
        settings=self.guidance;records=[]
        for piece,duration in enumerate(times):
            for fraction in np.linspace(0.,1.,settings.samples_per_piece):
                bases=np.stack([polynomial_basis_matrix(np.array([fraction*duration]),d)[0] for d in range(5)])
                state=bases@coeff[piece];cost,grad,contact=self._clearance_samples(state[0],state[2]);residual,jac=self._physical_residuals(state)
                records.append((piece,fraction,bases,state,cost,grad,bool(any(contact)),residual,jac,contact))
        collision=np.array([r[6] for r in records]);safety_values=np.array([sum(r[4]) for r in records]);safe,weights,collided=focused_safety(safety_values,collision,beta=settings.safety_beta)
        coefficient_gradient=np.zeros_like(coeff);time_gradient=np.zeros(2);detached_c=np.zeros_like(coeff);detached_t=np.zeros(2)
        def accumulate(row,g,weight=1.,detached=False):
            piece,fraction,bases,state=row[:4];gc=detached_c if detached else coefficient_gradient;gt=detached_t if detached else time_gradient
            gc[piece]+=weight*np.einsum('dk,dc->kc',bases[:4],g);gt[piece]+=weight*fraction*np.sum(g*state[1:5])
        measured={'environment':0.,'self':0.,'physical':0.,'curvature':0.,'acceleration':0.}
        for row,weight in zip(records,weights):
            accumulate(row,row[5][0]+row[5][1],weight)
        # LogSumExp is not additive; allocate it with the same normalized weights
        # for diagnostics while preserving the exact total safety cost.
        weighted=np.einsum('n,nk->k',weights,np.array([r[4] for r in records]));total=weighted.sum()
        if total>0:measured['environment'],measured['self']=safe*weighted/total
        residual=np.array([r[7] for r in records]);indices=residual.argmax(0);peak=residual[indices,np.arange(residual.shape[1])]
        for k,index in enumerate(indices):
            if peak[k]>0:
                measured['physical']+=settings.feasibility_weight*peak[k]
                accumulate(records[index],records[index][8][k],settings.feasibility_weight)
        for row in records:
            v,a=row[3][1,:3],row[3][2,:3];value,g,det=curvature_cost_gradient(v,a,self.config.max_speed,self.config.max_acceleration,settings.lateral_acceleration_ratio)
            weight=settings.feasibility_weight*settings.curvature_weight/len(records);measured['curvature']+=weight*value
            grad=np.zeros((4,8));grad[1:3,:3]=g;accumulate(row,grad,weight)
            grad[1:3,:3]=det;accumulate(row,grad,weight,detached=True)
        scale=np.r_[np.full(3,self.config.position_scale),self.config.yaw_scale,self.config.joint_scales]
        energy,gc,gt=acceleration_energy(coeff,times,settings.acceleration_weight/scale**2)
        measured['acceleration']=energy;coefficient_gradient+=gc;time_gradient+=gt
        contact_components={name:bool(any(r[9][i] for r in records)) for i,name in enumerate(('environment','self'))}
        self._term_result=(measured,collided,detached_c,detached_t,float(residual.max()),contact_components)
        return safe+sum(measured[k] for k in ('physical','curvature','acceleration')),coefficient_gradient,time_gradient

    def evaluate(self,obs,parameters):
        obs.validate_joints(self.robot.limits.joint_lower,self.robot.limits.joint_upper)
        if not np.isclose(obs.configuration[11],self.gripper_gap):raise ValueError('gripper mismatch')
        batch=np.asarray(parameters,dtype=float)
        if batch.ndim!=2 or batch.shape[1]!=34 or len(batch)==0 or not np.isfinite(batch).all():raise ValueError('parameters must be finite (K,34)')
        costs=[];gradients=[];detached=[];contacts=[];violations=[];components={};collision_components={'environment':[],'self':[]}
        for z in batch:
            minco,points,times,tail,frame=world_problem(z,obs);settings=self.guidance
            weights=np.asarray(self.config.jerk_weights)*settings.smoothness_weight
            cost,gp,gt,coeff,_,ge=evaluate_minco_objective(minco,points,times,terms=(self._sample_term,),energy_weights=weights,time_weight=settings.time_weight,boundary_gradients=True)
            measured,collided,dc,dt,violation,contact_components=self._term_result
            measured['jerk']=minco.jerk_energy(coeff,times,weights)[0];measured['time']=settings.time_weight*times.sum()
            goal,goal_gradient=goal_cost_gradient(obs.start_pva[0],tail[0],obs.goal,settings,collided=collided)
            ge[0]+=goal_gradient;cost+=goal;measured['progress']=goal
            _,dgp,dgt,_,_,dge=evaluate_minco_objective(minco,points,times,terms=(lambda t,c:(0.,dc,dt),),energy_weights=np.zeros(8),boundary_gradients=True)
            if collided:dge[0]+=goal_cost_gradient(obs.start_pva[0],tail[0],obs.goal,settings)[1]
            def pullback(gp,ge,gt):
                ge=ge.copy();ge[:,:3]=frame.to_local_vectors(ge[:,:3])
                return np.r_[frame.pullback_points8(gp).ravel(),ge.ravel(),gt]
            gradient=pullback(gp,ge,gt);excluded=pullback(dgp,dge,dgt)
            if not np.isfinite(cost) or not np.isfinite(gradient).all() or not np.isfinite(excluded).all():raise FloatingPointError('nonfinite MINCO guidance')
            costs.append(cost);gradients.append(gradient);detached.append(excluded);contacts.append(collided);violations.append(violation)
            for name,value in measured.items():components.setdefault(name,[]).append(value)
            for name,value in contact_components.items():collision_components[name].append(value)
        result=GuidanceBatch(np.asarray(costs),np.asarray(gradients),{k:np.asarray(v) for k,v in components.items()},np.asarray(violations),np.asarray(contacts),np.asarray(detached),{k:np.asarray(v) for k,v in collision_components.items()})
        self.last_result=result;return result
