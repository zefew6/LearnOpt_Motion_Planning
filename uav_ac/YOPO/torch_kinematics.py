"""Batched torch FK using only the public robot kinematic description."""
import torch
from torch.nn.functional import normalize


def flatness_rotation(acceleration,yaw,gravity=9.81):
    b3=normalize(torch.stack((-acceleration[...,0],-acceleration[...,1],gravity-acceleration[...,2]),-1),dim=-1,eps=1e-8)
    heading=torch.stack((yaw.cos(),yaw.sin(),torch.zeros_like(yaw)),-1)
    b2=normalize(torch.linalg.cross(b3,heading),dim=-1,eps=1e-8)
    b1=torch.linalg.cross(b2,b3)
    return torch.stack((b1,b2,b3),-1)


def axis_rotation(axis,angle):
    x,y,z=axis;zero=x*0
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
    eye=torch.eye(3,device=angle.device,dtype=angle.dtype)
    return eye+angle.sin()[...,None,None]*skew+(1-angle.cos())[...,None,None]*(skew@skew)


class TorchKinematics:
    def __init__(self,description):self.description=description
    def points(self,position,rotation,joints,gap,body_names,local_points):
        transforms={}
        for body in self.description['bodies']:
            if body['parent'] is None:
                transforms[body['name']]=(position,rotation);continue
            p,r=transforms[body['parent']]
            offset=position.new_tensor(body['position']);local=position.new_tensor(body['rotation'])
            for joint in body['joints']:
                axis=position.new_tensor(joint['axis']);pivot=position.new_tensor(joint['pivot'])
                angle=joints[...,joint['index']]-joint['reference'] if joint['index']<4 else position.new_tensor((gap-self.description['gripper_min'])/2-joint['reference'])
                if joint['kind']=='hinge':
                    motion=axis_rotation(axis,angle)
                    offset=offset+torch.einsum('ij,...j->...i',local,pivot-torch.einsum('...ij,j->...i',motion,pivot))
                    local=local@motion
                else:offset=offset+(local@axis)*angle
            transforms[body['name']]=(p+torch.einsum('...ij,...j->...i',r,offset),r@local)
        # Public geometry points are expressed in native body FLU; convert FRD.
        points=local_points*local_points.new_tensor([1.,-1.,-1.])
        unique=list(dict.fromkeys(body_names));columns={name:i for i,name in enumerate(unique)}
        pos=torch.stack([transforms[n][0] for n in unique],-2)
        rot=torch.stack([transforms[n][1] for n in unique],-3)
        indices=torch.tensor([columns[n] for n in body_names],device=position.device)
        return pos[...,indices,:]+torch.einsum('...sij,sj->...si',rot[...,indices,:,:],points)
