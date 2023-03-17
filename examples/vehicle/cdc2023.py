import numpy as np
from ReachMM import NeuralNetwork, NeuralNetworkControl, NeuralNetworkControlIF
from ReachMM.utils import plot_Y_X
from Vehicle import *
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from tabulate import tabulate
from ReachMM.utils import run_time, gen_ics_pert
from scipy.integrate import solve_ivp

device = 'cpu'
MC_N = 200

net = NeuralNetwork('models/100r100r2', device=device)

control = NeuralNetworkControl(net, device=device)
controlif = NeuralNetworkControlIF(net, mode='hybrid', method='CROWN', device=device)
# controlif = NeuralNetworkControlIFIBP(nn, device=device)
model = VehicleModel(control, controlif)
x0 = np.array([8,8,-2*np.pi/3,2])
pert = np.array([0.1,0.1,0.01,0.01])
# t_step = 0.01 if args.t_step is None else args.t_step
xlen = 4

t_step = 0.01
t_span = [0,1.5]
tt = np.arange(t_span[0],t_span[1]+t_step,t_step)

x0d = np.concatenate((x0 - pert,x0 + pert))

traj, runtime = run_time(model.compute_trajectory, x0, t_span, 'euler', t_step, enable_bar=False)
trajxx = traj(tt)
rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, 0, 0, 'euler', t_step, enable_bar=False)
# print(rs([0,1.25]))


X0 = gen_ics_pert(x0, pert, MC_N)
trajs = []
for mc_x0 in X0 :
    trajs.append(model.compute_trajectory(mc_x0, t_span, 'euler', t_step, enable_bar=False))


# eps, max_primer_depth, max_depth, cd, id, check_cont
# experiments = (((0,0,0,0,0,0.5), (0,0,0,0,1,0.5), (1,0,0,1,0,0.5), (0,0,0,2,0,0.5)),
#                ((1,0,1,0,0,0.5), (0.5,0,1,0,0,0.5), (0.5,1,1,0,0,0.5), (0.5,2,2,0,0,0.5)))
experiments = (((0,0,0,0,0,0.5,0,False), (  0,0,0,1,1,0.5,0,False), (  1,0,0,2,0,0.5,0,False), (   0,0,0,1,0,0.5,0,False)),
               ((1,1,1,0,0,0,0,False), (1,1,2,0,0,0,0,False), (0.5,1,2,0,0,0,0,False), (1,2,2,0,0,0,0,False)))

fig, axs = plt.subplots(2,4,dpi=100,figsize=[14,8],squeeze=False)
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.125, top=0.9, wspace=0.125, hspace=0.2)

for i, exps in enumerate(experiments) :
    for j, exp in enumerate(exps) :
        # eps, max_primer_depth, max_depth, cd, id, check_contr = exp
        eps, max_primer_depth, max_depth, cd, id, check_contr, dist, cut_dist = exp

        # rs = model.compute_reachable_set_eps(x0d, t_span, cd, id, 'euler', t_step, eps, max_depth, check_cont, False)
        # rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, enable_bar=False)
        rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, cut_dist, False, enable_bar=True)
        # rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, cd, id, 'euler', t_step, True, enable_bar=True)
        # print(runtime)


        # axs[0,i].set_title(f'$d^H$, $D_a={(2**cd)**xlen}$, $D_s={(2**id)**xlen}$',fontdict=dict(fontsize=20))
        axs[i,j].text(-0.5,8.5,f'runtime:\n${runtime:.3f}$',fontsize=15,verticalalignment='top')

        for n in range(MC_N) :
            # plot_Y_X(fig, axs[i,j], tt, trajs[n](tt)[0,:], trajs[n](tt)[1,:], xlim=[-1,9],ylim=[-1,9],lw=0.5)
            axs[i,j].plot(trajs[n](tt)[0,:], trajs[n](tt)[1,:], color='tab:red', zorder=0)
            axs[i,j].set_xlim([-1,9]); axs[i,j].set_ylim([-1,9])

        # rs.draw_sg_boxes(axs[i,j],np.arange(t_span[0],t_span[1],0.05))
        rs.draw_sg_boxes(axs[i,j],tt[:-1])
        # rs.draw_sg_boxes(axs[i,j],np.arange(t_span[0],t_span[1]+0.25,0.25))
        # print(rs([0,1.25]))


plt.show()