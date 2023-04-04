from XYDoubleIntegrator import *
import numpy as np
from ReachMM import NeuralNetwork, NeuralNetworkControl, NeuralNetworkControlIF
from ReachMM import ConstantDisturbance, ConstantDisturbanceIF
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from tabulate import tabulate
from ReachMM.utils import run_time, plot_Y_X
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn

device = 'cpu'

T = np.array([[0.5,0,0.5,0],[0,0.5,0,0.5],[0.5,0,-0.5,0],[0,0.5,0,-0.5]])
Ti  = np.array([[1,0,1,0],[0,1,0,1],[1,0,-1,0],[0,1,0,-1]])
st = nn.Linear(4,4,bias=False)
with torch.no_grad() :
    st.weight = nn.Parameter(torch.Tensor(Ti), requires_grad=False)
print(st)
print(st.weight)

net = nn.Sequential(
    st,
    NeuralNetwork('models/10r10r2')
)
print(net)

control = NeuralNetworkControl(net, u_len=2, device=device)
controlif = NeuralNetworkControlIF(net, u_len=2, mode='hybrid', method='CROWN', device=device)

model = RotXYDoubleIntegratorModel(control, controlif)

x0 = np.array([7,7,2,-4])
# pert = np.array([0.001,0.001,0.001,0.001])
pert = np.array([0.1,0.1,0.01,0.01])
xlen = 4

t_step = 0.01
t_span = [0,3]
tt = np.arange(t_span[0],t_span[1]+t_step,t_step)

s = T.shape[0]
mat = np.zeros((s*2,s*2))
mat[:s,:s] = np.clip(T,0,np.inf)
mat[s:,:s] = np.clip(T,-np.inf,0)
mat[:s,s:] = np.clip(T,-np.inf,0)
mat[s:,s:] = np.clip(T,0,np.inf)
x0d = mat @ np.concatenate(((x0 - pert),(x0 + pert)))
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
experiments = (((0,0,0,0,0,0.5,0,False), (  0,0,0,0,1,0.5,0,False), (  1,0,0,1,0,0.5,0,False), (   0,0,0,1,1,0.5,0,False)),
               ((1,0,1,0,0,0.5,0,False), (0.5,0,1,0,0,0.5,0,False), (0.5,1,1,0,0,0.5,0,False), (0.25,1,2,0,0,0.1,0,False)))

fig, axs = plt.subplots(2,4,dpi=100,figsize=[14,8],squeeze=False)
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.125, top=0.9, wspace=0.125, hspace=0.2)

for i, exps in enumerate(experiments) :
    for j, exp in enumerate(exps) :
        eps, max_primer_depth, max_depth, cd, id, check_contr, dist, cut_dist = exp
        # model.disturbance_if = ConstantDisturbanceIF([-dist],[dist])

        # rs = model.compute_reachable_set_eps(x0d, t_span, cd, id, 'euler', t_step, eps, max_depth, check_cont, False)
        rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, cut_dist, enable_bar=True)
        # print(runtime)

        plot_Y_X(fig, axs[i,j], tt, trajxx[0,:], trajxx[1,:],xlim=[-1,9],ylim=[-1,9])
        # plot_Y_X(fig, axs[i,j], tt, traj1(tt)[0,:], traj1(tt)[1,:],xlim=[-1,9],ylim=[-1,9])
        # plot_Y_X(fig, axs[i,j], tt, traj2(tt)[0,:], traj2(tt)[1,:],xlim=[-1,9],ylim=[-1,9])

        # axs[0,i].set_title(f'$d^H$, $D_a={(2**cd)**xlen}$, $D_s={(2**id)**xlen}$',fontdict=dict(fontsize=20))
        axs[i,j].text(-0.5,8.5,f'runtime:\n${runtime:.3f}$',fontsize=15,verticalalignment='top')
        rs.draw_sg_boxes(axs[i,j],np.arange(t_span[0],t_span[1]+0.05,0.05),Ti=Ti)
        # print(rs([0,1.25]))

plt.show()

