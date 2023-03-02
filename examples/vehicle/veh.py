import numpy as np
from ReachMM import NeuralNetworkControl, NeuralNetworkControlIF, NeuralNetworkControlIFIBP
from VehicleNeuralNetwork import VehicleNeuralNetwork
from VehicleModel import VehicleModel
from VehicleUtils import *
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from tabulate import tabulate
from ReachMM.utils import run_time
from scipy.integrate import solve_ivp

device = 'cpu'

nn = VehicleNeuralNetwork('twoobs-nost', device)

control = NeuralNetworkControl(nn, device=device)
controlif = NeuralNetworkControlIF(nn, mode='hybrid', method='CROWN', device=device)
# controlif = NeuralNetworkControlIFIBP(nn, device=device)
model = VehicleModel(control, controlif, u_step=0.25)
x0 = np.array([8,8,-2*np.pi/3,2])
pert = np.array([0.1,0.1,0.01,0.01])
# t_step = 0.01 if args.t_step is None else args.t_step
xlen = 4

t_step = 0.01
t_span = [0,1.25]
tt = np.arange(t_span[0],t_span[1]+t_step,t_step)

x0d = np.concatenate((x0 - pert,x0 + pert))

traj, runtime = run_time(model.compute_trajectory, x0, t_span,  enable_bar=False)
trajxx = traj(tt)
rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, 0, 0, 'euler', t_step, enable_bar=False)
# print(rs([0,1.25]))

# eps, max_primer_depth, max_depth, cd, id, check_cont
experiments = (((0,0,0,0,0,0.5), (0,0,0,0,1,0.5), (1,0,0,1,0,0.5), (0,0,0,2,0,0.5)),
               ((1,0,1,0,0,0.5), (0.5,0,1,0,0,0.5), (0.5,1,1,0,0,0.5), (0.5,2,2,0,0,0.5)))

fig, axs = plt.subplots(2,4,dpi=100,figsize=[14,8],squeeze=False)
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.125, top=0.9, wspace=0.125, hspace=0.2)

for i, exps in enumerate(experiments) :
    for j, exp in enumerate(exps) :
        eps, max_primer_depth, max_depth, cd, id, check_contr = exp

        # rs = model.compute_reachable_set_eps(x0d, t_span, cd, id, 'euler', t_step, eps, max_depth, check_cont, False)
        rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, enable_bar=False)
        # print(runtime)

        plot_Y_X(fig, axs[i,j], tt, trajxx[0,:], trajxx[1,:],xlim=[-1,9],ylim=[-1,9])

        # axs[0,i].set_title(f'$d^H$, $D_a={(2**cd)**xlen}$, $D_s={(2**id)**xlen}$',fontdict=dict(fontsize=20))
        axs[i,j].text(-0.5,8.5,f'runtime:\n${runtime:.3f}$',fontsize=15,verticalalignment='top')
        rs.draw_sg_boxes(axs[i,j],np.arange(t_span[0],t_span[1]+0.05,0.05))
        # print(rs([0,1.25]))

plt.show()