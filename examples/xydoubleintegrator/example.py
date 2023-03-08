from XYDoubleIntegrator import *
import numpy as np
from ReachMM import NeuralNetwork, NeuralNetworkControl, NeuralNetworkControlIF
from ReachMM import ConstantDisturbance, ConstantDisturbanceIF
from ReachMM import LinearControl, LinearControlIF
from ReachMM.decomp import d_positive
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from tabulate import tabulate
from ReachMM.utils import run_time, plot_Y_X
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
from numpy.linalg import inv
from control import lqr

device = 'cpu'
u_step = 0.01

A = np.array([ [0,0,1,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]])
B = np.array([ [0,0], [0,0], [1,0], [0,1] ])
Q = np.array([ [1,0,0,0], [0,1,0,0], [0,0,2,0], [0,0,0,2] ])
R = np.array([ [0.5,0], [0,0.5] ])

K, S, E = lqr(A, B, Q, R)

W, V = np.linalg.eig(A - B@K)
print(V)
# Ti = np.empty((4,4))
# Ti[:,0] = np.real(V[:,0])
# Ti[:,1] = np.imag(V[:,0])
# Ti[:,2] = np.real(V[:,2])
# Ti[:,3] = np.imag(V[:,2])
Ti = V
print(Ti)
# Ti = np.eye(4)
T = inv(Ti)
# print(Ti)
# print(T)
print(K)
print(T@(A-B@K)@Ti)

dT = d_positive(T)
dTi = d_positive(Ti)
st = nn.Linear(4,4,bias=False)
with torch.no_grad() :
    st.weight = nn.Parameter(torch.Tensor(Ti), requires_grad=False)

net = nn.Sequential(
    st,
    NeuralNetwork('models/10r10r2')
)
# net = NeuralNetwork('models/10r10r2')

# control = NeuralNetworkControl(net, u_len=2, device=device)
# controlif = NeuralNetworkControlIF(net, u_len=2, mode='hybrid', method='CROWN', device=device)
control = LinearControl(np.zeros_like(K))
controlif = LinearControlIF(np.zeros_like(K), 'local')
# control = LinearControl(-T@B@K@Ti)
# controlif = LinearControlIF(-T@B@K@Ti, 'local')

model = XYDoubleIntegratorModel(control, controlif, u_step=u_step, T=T)
model.A = T@(A-B@K)@Ti
model.Am, model.An = d_metzler(model.A, True)
# model.B = np.eye(4)
# model.Bp, model.Bn = d_positive(model.B, True)

x0 = np.array([7,7,2,-4])
# x0 = np.array([2,2,0,0])
# pert = np.array([0.001,0.001,0.001,0.001])
pert = np.array([0.1,0.1,0.01,0.01])
# pert = np.array([0.5,0.5,0.1,0.1])
# pert = np.array([0.7,0.7,0.7,0.7])
xlen = 4

t_step = 0.01
t_span = [0,2]
tt = np.arange(t_span[0],t_span[1]+t_step,t_step)

x0d = np.concatenate((x0 - pert, x0 + pert))
x0d = dT @ x0d
print(x0d)

traj, runtime = run_time(model.compute_trajectory, T@x0, t_span, 'euler', t_step, enable_bar=False)
# model.disturbance = ConstantDisturbance([0.05])
# traj1, runtime = run_time(model.compute_trajectory, x0-pert, t_span, 'euler', t_step, enable_bar=False)
# model.disturbance = ConstantDisturbance([-0.05])
# traj2, runtime = run_time(model.compute_trajectory, x0+pert, t_span, 'euler', t_step, enable_bar=False)
trajxx = Ti @ traj(tt)
rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, 0, 0, 'euler', t_step, enable_bar=False)
# print(rs([0,1.25]))

# eps, max_primer_depth, max_depth, cd, id, check_cont, dist, cut_dist
# experiments = (((0,0,0,0,0,0.5,0.05,False), (  0,0,0,0,1,0.5,0.05,True), (  0,0,0,0,2,0.5,0.05,True), (   0,0,0,1,1,0.5,0.1,True)),
#                ((1,0,1,0,0,0.5,0.1,True), (0.5,0,1,0,0,0.5,0.1,True), (0.5,1,1,0,0,0.5,0.1,True), (0.25,1,2,0,0,0.1,0.1,True)))
experiments = (((0,0,0,0,0,0.5,0,False), (  0,0,0,0,1,0.5,0,False), (  1,0,0,1,0,0.5,0,False), (   0,0,0,2,0,0.5,0,False)),
               ((1,0,1,0,0,0.5,0,False), (0.5,0,1,0,0,0.5,0,False), (0.5,1,1,0,0,0.5,0,False), (0.25,1,2,0,0,0.1,0,False)))

fig, axs = plt.subplots(2,4,dpi=100,figsize=[14,8],squeeze=False)
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.125, top=0.9, wspace=0.125, hspace=0.2)

for i, exps in enumerate(experiments) :
    for j, exp in enumerate(exps) :
        eps, max_primer_depth, max_depth, cd, id, check_contr, dist, cut_dist = exp
        model.disturbance_if = ConstantDisturbanceIF([-dist],[dist])

        # rs = model.compute_reachable_set_eps(x0d, t_span, cd, id, 'euler', t_step, eps, max_depth, check_cont, False)
        rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, cut_dist, enable_bar=True)
        # print(runtime)

        plot_Y_X(fig, axs[i,j], tt, trajxx[0,:], trajxx[1,:],xlim=[-1,9],ylim=[-1,9])
        # plot_Y_X(fig, axs[i,j], tt, traj1(tt)[0,:], traj1(tt)[1,:],xlim=[-1,9],ylim=[-1,9])
        # plot_Y_X(fig, axs[i,j], tt, traj2(tt)[0,:], traj2(tt)[1,:],xlim=[-1,9],ylim=[-1,9])

        # axs[0,i].set_title(f'$d^H$, $D_a={(2**cd)**xlen}$, $D_s={(2**id)**xlen}$',fontdict=dict(fontsize=20))
        axs[i,j].text(-0.5,8.5,f'runtime:\n${runtime:.3f}$',fontsize=15,verticalalignment='top')
        rs.draw_sg_boxes(axs[i,j],np.arange(t_span[0],t_span[1]+0.05,0.05),T=Ti)
        print(rs(2))

plt.show()

