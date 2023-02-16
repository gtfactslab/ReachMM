import numpy as np
from ReachMM import MixedMonotoneModel, NeuralNetworkControl, NeuralNetworkControlIF
from ReachMM.decomp import d_b1b2, d_x2
import torch
import torch.nn as nn
from ModuleFromTxt import ModuleFromTxt
import matplotlib.pyplot as plt
import time

# device = 'cpu'
# a = 2*0.7523303216858767
a = 1

def run_time (func, *args, **kwargs) :
    before = time.time()
    ret = func(*args, **kwargs)
    after = time.time()
    return ret, (after - before)

class B1Model (MixedMonotoneModel) :
    def __init__(self, control=None, control_if=None, u_step=0.2):
        super().__init__(control, control_if, u_step)
    def f (self, x, u) :
        return np.array([x[1], u[0]*x[1]**2 - x[0]])
    def d (self, x, u, xh, uh) :
        # print(x, xh)
        # d2 = u[0]*(d_x2(x[1], xh[1]) if u[0] > 0 else d_x2(xh[1],x[1])) - xh[0]
        ret = np.array([
            x[1],
            u[0]*x[1]**2 - xh[0]
        ])
        return ret

class B1TModel (MixedMonotoneModel) :
    def __init__(self, control=None, control_if=None, u_step=0.1):
        super().__init__(control, control_if, u_step)
    def f (self, x, u) :
        return np.array([
            u[0]*x[0]**2 + x[1] - a*x[0],
            a*u*x[0]**2 + (-1 - a**2)*x[0] + a*x[1]
        ])
    def d (self, x, u, xh, uh) :
        return np.array([
            u[0]*x[0]**2 + x[1] - a*x[0],
            a*u*x[0]**2 + (-1 - a**2)*xh[0] + a*x[1]
        ])

# nn_model = ModuleFromTxt('nn_1_relu_tanh')
# print(nn_model)
# print(nn_model.layers[0])

Tnp = np.array([[0,1.0],[-1.0,a]])
# Tinp = np.array([[0,-1.0],[1.0,0]])
Tinp = np.linalg.inv(Tnp)

T = nn.Linear(2,2,bias=False)
Ti = nn.Linear(2,2,bias=False)
with torch.no_grad() :
    T.weight = nn.Parameter(torch.Tensor(Tnp), requires_grad=False)
    Ti.weight = nn.Parameter(torch.Tensor(Tinp), requires_grad=False)
    # T.weight = nn.Parameter(torch.tensor([[0,1.0],[-1.0,0]]), requires_grad=False)
    # Ti.weight = nn.Parameter(torch.tensor([[0,-1.0],[1.0,0]]), requires_grad=False)
print(T.weight)
print(Ti.weight)

nn_model_pre = ModuleFromTxt('nn_1_relu_tanh')

nn_model = nn.Sequential(
    Ti,
    nn_model_pre
)

control = NeuralNetworkControl(nn_model_pre, u_len=1)
controlif = NeuralNetworkControlIF(nn_model, mode='hybrid', x_len=2, method='CROWN', u_len=1)
model = B1Model (control, controlif)
Tmodel = B1TModel (control, controlif)


x0 = np.array([0.85, 0.55])
print(nn_model_pre(torch.Tensor(x0)))
eps = np.array([0.05, 0.05])
# x0d = np.concatenate((x0 - eps,x0 + eps))
x0d = np.empty(len(x0)*2)
x0d[:len(x0d)//2] = np.min((Tnp@(x0 - eps),Tnp@(x0 + eps)),axis=0)
x0d[len(x0d)//2:] = np.max((Tnp@(x0 - eps),Tnp@(x0 + eps)),axis=0)
print(Tnp@(x0 - eps))
print(Tnp@(x0 + eps))
print(x0d)

t_step = 0.01
t_end = 0.2*10
t_span = [0,t_end]

traj, runtime = run_time(model.compute_trajectory, x0, t_span, t_step, embed=False, enable_bar=False)
print(f'One real trajectory: {runtime}')
tt = traj['t']
xx = traj['x']
uu = traj['u']
print(f'mean: {np.mean(uu)}')
dtraj, runtime = run_time(Tmodel.compute_trajectory, x0d, t_span, t_step, embed=True, enable_bar=False)
print(f'One embedded trajectory: {runtime}')
dtt = dtraj['t']
dxx = dtraj['x']
duu = dtraj['u']

fig, axs = plt.subplots(2,2,dpi=100,figsize=[10,10],squeeze=False)
axs[0,0].plot(xx[0,:],xx[1,:])
axs[0,0].set_xlim([-0.5,1.25]); axs[0,0].set_ylim([-1,1])
print(xx.shape)
print(dxx.shape)
dxx[:2,:] = Tinp @ dxx[:2,:]
dxx[2:,:] = Tinp @ dxx[2:,:]
axs[0,0].plot(xx[0,:],xx[1,:])
axs[0,0].plot(dxx[0,:],dxx[1,:])
axs[0,0].plot(dxx[2,:],dxx[3,:])

axs[0,1].plot(xx[0,:],xx[1,:])
axs[0,1].set_xlim([-0.5,1.25]); axs[0,1].set_ylim([-1,1])
axs[1,0].plot(xx[0,:],xx[1,:])
axs[1,0].set_xlim([-0.5,1.25]); axs[1,0].set_ylim([-1,1])
axs[1,1].plot(xx[0,:],xx[1,:])
axs[1,1].set_xlim([-0.5,1.25]); axs[1,1].set_ylim([-1,1])

rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, t_step, control_divisions=0, integral_divisions=0, enable_bar=False)
print(f'RS runtime: {runtime}')
rs.draw_sg_boxes(axs[0,0])

rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, t_step, control_divisions=0, integral_divisions=3, enable_bar=False, repartition=True)
print(f'RS runtime: {runtime}')
rs.draw_sg_boxes(axs[0,1])

rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, t_step, control_divisions=3, integral_divisions=0, enable_bar=False, repartition=False)
print(f'RS runtime: {runtime}')
rs.draw_sg_boxes(axs[1,0])

# rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, t_step, control_divisions=0, integral_divisions=3, enable_bar=False, repartition=False)
# print(f'RS runtime: {runtime}')
# rs.draw_sg_boxes(axs[1,1], T=Tinp)

plt.show()
