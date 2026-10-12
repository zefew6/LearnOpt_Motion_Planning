"""Four-piece MINCO with both lateral-jerk endpoint constraints eliminated."""
import numpy as np
from .contracts import finite_array
from .encoding import wrap
from ..trajectory.gcopter.minco import MINCOQuintic


class InteriorMap:
    def __init__(self,head_pva,tail_pva,durations=None):
        self.head=finite_array(head_pva,(3,8),'head_pva')
        tail=np.array(tail_pva,dtype=float,copy=True)
        if tail.shape==(3,8): tail[0,3]=self.head[0,3]+wrap(tail[0,3]-self.head[0,3])
        self.tail=finite_array(tail,(3,8),'tail_pva')
        self.durations=finite_array(np.full(4,.75) if durations is None else durations,(4,),'durations')
        if np.any(self.durations<=1e-5): raise ValueError('durations must be positive and not singular')
        self.minco=MINCOQuintic(self.head,self.tail,4)
        self.free_indices=np.array([i for i in range(24) if i not in (0,1,16,17)])
        self.dependent=np.array([0,1,16,17])
        zero=np.zeros((3,8));base=self._jerks(zero)
        response=[]
        for index in range(24):
            points=zero.copy();points.ravel()[index]=1.
            response.append(self._jerks(points)-base)
        matrix=np.array(response).T
        block=matrix[:,self.dependent]
        condition=np.linalg.cond(block)
        if not np.isfinite(condition) or condition>1e10:
            raise ValueError('endpoint jerk elimination is ill conditioned')
        self.condition_number=float(condition)
        self.offset=np.zeros(24);self.offset[self.dependent]=np.linalg.solve(block,-base)
        self.jacobian=np.zeros((24,20));self.jacobian[self.free_indices,np.arange(20)]=1.
        self.jacobian[self.dependent]=np.linalg.solve(block,-matrix[:,self.free_indices])

    def _jerks(self,points):
        coeff,_=self.minco.solve(points,self.durations);c=coeff.reshape(4,6,8)
        t=self.durations[-1]
        return np.r_[6*c[0,3,:2],6*c[-1,3,:2]+24*t*c[-1,4,:2]+60*t*t*c[-1,5,:2]]

    def decode(self,free):
        free=finite_array(free,(20,),'free')
        return (self.offset+self.jacobian@free).reshape(3,8)

    def pullback(self,gradient):
        gradient=finite_array(gradient,(3,8),'points_gradient')
        return self.jacobian.T@gradient.ravel()

    def seed(self):
        fractions=(np.cumsum(self.durations)[:-1]/np.sum(self.durations))[:,None]
        return (self.head[0]+fractions*(self.tail[0]-self.head[0])).ravel()[self.free_indices]
