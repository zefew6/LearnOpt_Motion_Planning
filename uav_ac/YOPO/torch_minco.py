"""Batched differentiable two-piece quintic MINCO; no iterative optimization."""
import math
import torch


def basis(t,order):
    return torch.stack([t**(k-order)*(math.factorial(k)/math.factorial(k-order)) if k>=order else torch.zeros_like(t) for k in range(6)],dim=-1)


def solve_minco(head,tail,middle,times):
    """Solve the12 boundary/continuity equations for all candidates at once."""
    zero=torch.zeros_like(times[...,0]);z6=zero[...,None].expand(*zero.shape,6)
    rows=[];rhs=[]
    for d in range(3):rows.append(torch.cat((basis(zero,d),z6),-1));rhs.append(head[...,d,:])
    rows.append(torch.cat((basis(times[...,0],0),z6),-1));rhs.append(middle)
    rows.append(torch.cat((z6,basis(zero,0)),-1));rhs.append(middle)
    for d in range(1,5):rows.append(torch.cat((basis(times[...,0],d),-basis(zero,d)),-1));rhs.append(torch.zeros_like(middle))
    for d in range(3):rows.append(torch.cat((z6,basis(times[...,1],d)),-1));rhs.append(tail[...,d,:])
    matrix=torch.stack(rows,-2);right=torch.stack(rhs,-2)
    return torch.linalg.solve(matrix,right).reshape(*times.shape[:-1],2,6,8)


def states(coeff,times,samples_per_piece):
    fraction=torch.linspace(0,1,samples_per_piece,device=times.device,dtype=times.dtype)
    t=times[...,None]*fraction
    samples=torch.stack([torch.einsum('...psk,...pkc->...psc',basis(t,d),coeff) for d in range(4)],-2)
    return samples.flatten(-4,-3)


def energy(coeff,times,order,weights):
    n=6-order
    power=torch.arange(n,device=times.device,dtype=times.dtype)
    multiplier=times.new_tensor([math.factorial(k)/math.factorial(k-order) for k in range(order,6)])
    derivative=coeff[...,order:,:]*multiplier[:,None]
    exponent=power[:,None]+power[None,:]+1
    gram=times[...,None,None]**exponent/exponent
    return (torch.einsum('...pic,...pij,...pjc->...pc',derivative,gram,derivative)*weights).sum((-2,-1))
