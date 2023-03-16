from ReachMM import NeuralNetworkControl, NeuralNetworkControlIF
from ReachMM import NeuralNetwork
from ReachMM.utils import run_time, gen_ics_pert
from DoubleIntegrator import *
import torch
import matplotlib.pyplot as plt
import numpy as np

# device = 'cuda:0'
device = 'cpu'
netpath = '10r5r1'
MC_N = 200

t_step = 1
t_span = [0,6]
# tt = np.arange(t_span[0],t_span[1] + t_step,t_step)
tt = np.arange(t_span[1])
print(tt)

net = NeuralNetwork('models/' + netpath)
print(f'loaded net from models/{netpath}')

control = NeuralNetworkControl(net,device=device)
# controlif = NeuralNetworkControlIF(net, mode='hybrid', method='CROWN', device=device)
controlif = NeuralNetworkControlIF(net, mode='disclti', method='CROWN', device=device)

model = DoubleIntegratorModel(control, controlif, u_step=1)

x0 = np.array([2.75, 0])
pert = np.array([0.25,0.25])

x0d = np.concatenate((x0 - pert, x0 + pert))

traj, runtime = run_time(model.compute_trajectory, x0, t_span, 'euler', t_step, enable_bar=False)
rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, 3, 0, 'euler', t_step, enable_bar=False)

print(runtime)
print(rs(6))

# eps, max_primer_depth, max_depth, cd, id, check_cont, dist, cut_dist
experiments = (((0,0,0,0,0,0.5,0,False), (  0,0,0,4,0,0.5,0,False), (  1,0,0,2,2,0.5,0,False), (   0,0,0,3,2,0.5,0,False)),
               ((0.1,1,3,0,0,0,0,False), (0.1,2,4,0,0,0,0,False), (0.1,2,10,0,0,0,0,False), (0.07,2,10,0,0,0,0,False)))

fig, axs = plt.subplots(2,4,dpi=100,figsize=[14,8],squeeze=False)
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.125, top=0.9, wspace=0.125, hspace=0.2)

# RANGES = [
#     (x0 - pert),
#     ()
# ]
# X0 = gen_ics(np.column_stack((x0-pert,x0+pert)), MC_N)
print(x0-pert, x0+pert)
# X0 = np.random.uniform(x0-pert, x0+pert, MC_N)
X0 = gen_ics_pert(x0, pert, MC_N)
# print(X0)

trajs = []

for mc_x0 in X0 :
    trajs.append(model.compute_trajectory(mc_x0, t_span, 'euler', t_step, enable_bar=False))

# print(trajs[0](tt)[0,:])
# print(trajs[0](tt)[1,:])

for i, exps in enumerate(experiments) :
    for j, exp in enumerate(exps) :
        eps, max_primer_depth, max_depth, cd, id, check_contr, dist, cut_dist = exp

        # rs = model.compute_reachable_set_eps(x0d, t_span, cd, id, 'euler', t_step, eps, max_depth, check_cont, False)
        rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, cut_dist, enable_bar=True)
        print(runtime)

        axs[i,j].set_xlim([-2.5,3.5])
        axs[i,j].set_ylim([-1.5,1.5])

        # axs[0,i].set_title(f'$d^H$, $D_a={(2**cd)**xlen}$, $D_s={(2**id)**xlen}$',fontdict=dict(fontsize=20))
        axs[i,j].text(-2,1,f'runtime:\n${runtime:.3f}$',fontsize=15,verticalalignment='top')
        rs.draw_sg_boxes(axs[i,j],tt)
        # print(rs(tt))
        print('rs.get_max_depth(0)', rs.get_max_depth(1))

        for n in range(MC_N) :
            axs[i,j].scatter(trajs[n](tt)[0,:],trajs[n](tt)[1,:], s=0.25, c='r')

        # input()

plt.show()