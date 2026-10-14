"""Batched torch YOPO-MINCO guide with an explicit ESDF/sphere representation.

All predicted trajectory states, FK, collision costs and derivatives stay on
one torch device. No iterative optimization or per-state MuJoCo queries.
Sphere-cover/voxel distances are a training approximation, not exact geometry.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from .torch_minco import solve_minco,states,energy
from .torch_kinematics import TorchKinematics,flatness_rotation


def norm(x):return torch.linalg.vector_norm(x,dim=-1)

def huber(x):return torch.where(x.abs()<1,.5*x.square(),x.abs()-.5)


def body_rate_squared(a,j,yaw,yaw_rate,g):
    force=torch.stack((-a[...,0],-a[...,1],g-a[...,2]),-1)
    df=-j;length=norm(force).clamp_min(1e-8);b3=force/length[...,None]
    db3=(df-b3*(b3*df).sum(-1,keepdim=True))/length[...,None]
    heading=torch.stack((yaw.cos(),yaw.sin(),torch.zeros_like(yaw)),-1)
    dh=yaw_rate[...,None]*torch.stack((-yaw.sin(),yaw.cos(),torch.zeros_like(yaw)),-1)
    cross=torch.linalg.cross(b3,heading);dc=torch.linalg.cross(db3,heading)+torch.linalg.cross(b3,dh)
    length=norm(cross).clamp_min(1e-8);b2=cross/length[...,None]
    db2=(dc-b2*(b2*dc).sum(-1,keepdim=True))/length[...,None]
    b1=torch.linalg.cross(b2,b3);db1=torch.linalg.cross(db2,b3)+torch.linalg.cross(b2,db3)
    r=torch.stack((b1,b2,b3),-1);dr=torch.stack((db1,db2,db3),-1)
    skew=r.transpose(-1,-2)@dr
    return skew[...,2,1].square()+skew[...,0,2].square()+skew[...,1,0].square()


@dataclass
class TorchGuidanceBatch:
    costs: torch.Tensor
    collided: torch.Tensor
    violations: torch.Tensor
    components: dict
    collision_components: dict

    def single(self,index):
        return TorchGuidanceBatch(self.costs[index],self.collided[index],self.violations[index],
            {k:v[index] for k,v in self.components.items()},{k:v[index] for k,v in self.collision_components.items()})

    def numpy(self):
        from .privileged import GuidanceBatch
        def array(t):return t.detach().cpu().numpy()
        return GuidanceBatch(array(self.costs),np.empty((len(self.costs),34)),
            {k:array(v) for k,v in self.components.items()},array(self.violations),array(self.collided),
            collision_components={k:array(v) for k,v in self.collision_components.items()})


class TorchGuidance:
    def __init__(self,oracle,device,*,dtype=torch.float32):
        self.device=torch.device(device);self.dtype=dtype;self.oracle=oracle
        self.cfg=oracle.config;self.settings=oracle.guidance;self.gap=oracle.gripper_gap
        self.robot=oracle.robot;self.quad=oracle.evaluator.quad;self.mass=oracle.evaluator.mass
        self.kin=TorchKinematics(self.robot.kinematic_description())
        e=oracle.evaluator;self.names=e.sphere_names
        self.local=self.tensor(e.sphere_points);self.radii=self.tensor(e.sphere_radii)
        self.pairs=torch.tensor(e.self_pairs,device=self.device,dtype=torch.long)
        esdf=e.esdf
        self.grid=self.tensor(esdf.values).permute(2,1,0)[None,None].contiguous()
        self.origin=self.tensor(esdf.origin);self.upper=self.tensor(esdf.upper)
        self.grid_margin=np.sqrt(3.)*esdf.resolution
        self.lower_joints=self.tensor(self.robot.limits.joint_lower);self.upper_joints=self.tensor(self.robot.limits.joint_upper)
        self.bounds=self.tensor(e.bounds)

    def tensor(self,values):return torch.tensor(np.array(values,copy=True),device=self.device,dtype=self.dtype)

    def physical_residuals(self,values):
        s,v,a,j=values.unbind(-2);cfg=self.cfg;q=self.quad
        rows=[norm(v[...,:3])/cfg.max_speed-1,norm(a[...,:3])/cfg.max_acceleration-1,norm(j[...,:3])/self.settings.jerk_max-1,
            v[...,3].abs()/cfg.max_yaw_rate-1,a[...,3].abs()/cfg.max_yaw_acceleration-1]
        for i in range(4):
            k=i+4
            rows.extend(((s[...,k]-self.upper_joints[i])/.8,(self.lower_joints[i]-s[...,k])/.8,
                v[...,k].abs()/cfg.joint_velocity_limits[i]-1,a[...,k].abs()/cfg.joint_acceleration_limits[i]-1))
        force=torch.stack((-a[...,0],-a[...,1],q.g-a[...,2]),-1);length=norm(force).clamp_min(1e-8);thrust=self.mass*length;maximum=4*q.max_thrust
        cosine=(force[...,2]/length).clamp(-1+1e-7,1-1e-7)
        rows.extend(((thrust-maximum)/maximum,(4*q.min_thrust-thrust)/maximum,
            torch.acos(cosine)/q.max_tilt_angle-1,
            body_rate_squared(a[...,:3],j[...,:3],s[...,3],v[...,3],q.g)/cfg.max_body_rate**2-1))
        for sign,bound in ((1,self.bounds[1]),(-1,self.bounds[0])):
            rows.extend(sign*(s[...,i]-bound[i])/.5 for i in range(3))
        return torch.stack(rows,-1)

    def clearances(self,values):
        sigma=values[...,0,:];acceleration=values[...,2,:3]
        positions=self.kin.points(sigma[...,:3],flatness_rotation(acceleration,sigma[...,3],self.quad.g),sigma[...,4:],self.gap,self.names,self.local)
        normalized=2*(positions-self.origin)/(self.upper-self.origin)-1
        query=normalized.clamp(-1,1).reshape(1,1,1,-1,3)
        distance=F.grid_sample(self.grid,query,mode='bilinear',padding_mode='border',align_corners=True).reshape(positions.shape[:-1])
        # Bound the representation even when a candidate leaves the stored grid.
        boundary=torch.minimum(positions-self.origin,self.upper-positions).amin(-1)
        environment=(torch.minimum(distance,boundary)-self.radii).amin(-1)
        if len(self.pairs):
            # Min reduction retains only the closest pair in backward. Scalar
            # gripper-opening exclusions are inherited from public geometry.
            first,second=self.pairs.unbind(-1)
            self_gap=(norm(positions[...,first,:]-positions[...,second,:])-self.radii[first]-self.radii[second]).amin(-1)
        else:self_gap=torch.full_like(environment,1e3)
        return environment,self_gap

    def _world(self,parameters,observations):
        p=parameters.double();head=torch.tensor(np.stack([o.start_pva for o in observations]),device=p.device,dtype=p.dtype)[:,None].expand(-1,p.shape[1],-1,-1)
        yaw=head[...,0,3];c=yaw.cos();s=yaw.sin();zero=torch.zeros_like(c);one=torch.ones_like(c)
        r=torch.stack((c,-s,zero,s,c,zero,zero,zero,one),-1).reshape(*c.shape,3,3)
        def transform(v):return torch.einsum('...ij,...j->...i',r,v)
        middle=torch.cat((transform(p[...,:3])+head[...,0,:3],p[...,3:4]+yaw[...,None],p[...,4:8]),-1)
        local=p[...,8:32].reshape(*p.shape[:-1],3,8)
        xyz=torch.einsum('...ij,...dj->...di',r,local[...,:3])+torch.stack((head[...,0,:3],torch.zeros_like(head[...,0,:3]),torch.zeros_like(head[...,0,:3])),-2)
        angular=local[...,3:4]+torch.stack((yaw,zero,zero),-1)[...,None]
        tail=torch.cat((xyz,angular,local[...,4:]),-1)
        return head,tail,middle,p[...,-2:]

    def evaluate(self,parameters,observations,*,detach_guidance=True):
        head,tail,middle,times=self._world(parameters,observations)
        coeff=solve_minco(head,tail,middle,times)
        values=states(coeff,times,self.settings.samples_per_piece).to(self.dtype)
        environment,own=self.clearances(values);settings=self.settings
        def barrier(gap,margin,origin,decay,weight):
            raw=torch.exp(((origin-gap)/decay).clamp(-700,50))-np.exp((origin-margin)/decay)
            return weight*raw.clamp_min(0)
        # Voxel uncertainty increases the safety penalty, not the contact flag.
        env_cost=barrier(environment-self.grid_margin,settings.environment_safe,settings.environment_d0,settings.environment_decay,settings.safety_weight)
        self_cost=barrier(own,settings.self_safe,0.,settings.self_decay,settings.self_weight)
        contacts=(environment<=0)|(own<=0);collided=contacts.any(-1)
        first=contacts.to(torch.int64).argmax(-1);index=(first-1).clamp_min(0)
        safety=env_cost+self_cost
        focused=safety.gather(-1,index[...,None]).squeeze(-1)
        lse=(torch.logsumexp(settings.safety_beta*safety,-1)-np.log(safety.shape[-1]))/settings.safety_beta
        safety=torch.where(collided,focused,lse)
        residual=self.physical_residuals(values)
        physical=settings.feasibility_weight*residual.amax(-2).clamp_min(0).sum(-1)
        v=values[...,1,:3];a=values[...,2,:3];speed=norm(v)
        curvature=norm(torch.linalg.cross(v,a))/speed.clamp_min(.5).pow(3)
        cap=torch.sqrt(settings.lateral_acceleration_ratio*self.cfg.max_acceleration/curvature.clamp_min(1e-6)).clamp(.5,self.cfg.max_speed)
        if detach_guidance:cap=cap.detach()
        curvature_cost=settings.feasibility_weight*settings.curvature_weight*((speed-cap).clamp_min(0)/self.cfg.max_speed).mean(-1)
        jerk=energy(coeff,times,3,coeff.new_tensor(self.cfg.jerk_weights)*settings.smoothness_weight)
        scale=coeff.new_tensor([self.cfg.position_scale]*3+[self.cfg.yaw_scale]+list(self.cfg.joint_scales))
        acceleration=energy(coeff,times,2,settings.acceleration_weight/scale.square())
        time=settings.time_weight*times.sum(-1)
        goal=torch.tensor(np.stack([o.goal for o in observations]),device=self.device,dtype=tail.dtype)[:,None]
        direction=goal[...,:3]-head[...,0,:3];direction=direction/norm(direction).clamp_min(1e-6)[...,None]
        error=tail[...,0,:3]-goal[...,:3];parallel=(error*direction).sum(-1,keepdim=True)*direction
        yaw_error=tail[...,0,3]-goal[...,3];yaw_error=torch.atan2(yaw_error.sin(),yaw_error.cos())
        progress=settings.goal_weight*(huber(parallel).sum(-1)+settings.goal_perpendicular_weight*huber(error-parallel).sum(-1)+settings.goal_pose_weight*(huber(yaw_error/.7)+huber((tail[...,0,4:]-goal[...,4:])/.8).sum(-1)))
        if detach_guidance:progress=torch.where(collided,progress.detach(),progress)
        costs=safety+physical+curvature_cost+jerk+acceleration+time+progress
        return TorchGuidanceBatch(costs,collided,residual.amax((-2,-1)),
            dict(safety=safety,physical=physical,curvature=curvature_cost,jerk=jerk,acceleration=acceleration,time=time,progress=progress),
            dict(environment=(environment<=0).any(-1),self=(own<=0).any(-1)))


class TorchGuidanceOracle:
    """Float64 CPU reference adapter for the existing FD/neural audit tools."""
    def __init__(self,oracle):
        self.backend=TorchGuidance(oracle,'cpu',dtype=torch.float64)
        self.robot=oracle.robot;self.evaluator=oracle.evaluator;self.config=oracle.config
        self.last_result=None

    @torch.enable_grad()
    def evaluate(self,obs,parameters):
        from .privileged import GuidanceBatch
        z=torch.tensor(np.asarray(parameters),dtype=torch.float64,requires_grad=True)[None]
        result=self.backend.evaluate(z,[obs]);gradient=torch.autograd.grad(result.costs.sum(),z)[0][0]
        full=self.backend.evaluate(z,[obs],detach_guidance=False)
        full_gradient=torch.autograd.grad(full.costs.sum(),z)[0][0]
        def array(t):return t.detach().numpy()
        measured=result.single(0)
        self.last_result=GuidanceBatch(array(measured.costs),array(gradient),{k:array(v) for k,v in measured.components.items()},
            array(measured.violations),array(measured.collided),array(full_gradient-gradient),
            {k:array(v) for k,v in measured.collision_components.items()})
        return self.last_result
