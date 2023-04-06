import argparse

parser = argparse.ArgumentParser(description="Double Integrator Experiments for CDC Paper")

parser.add_argument('-T', '--t_end', help="End time for simulation",\
                    type=float, default=1.25)
parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",\
                    type=int, default=1)
parser.add_argument('--table', help="Enable for Table I", \
                    default=False, action='store_true')

args = parser.parse_args()

import numpy as np
from ReachMM import NeuralNetwork, NeuralNetworkControl, NeuralNetworkControlIF
from ReachMM.utils import plot_Y_X
from ReachMM.reach import volume
from Vehicle import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
from tqdm import tqdm
from tabulate import tabulate
from ReachMM.utils import run_time, gen_ics_pert
from scipy.integrate import solve_ivp
import torch

runtime_N = args.runtime_N

EXPERIMENT = 2
u_step = 0.125
# u_step = 0.1

device = 'cpu'
MC_N = 200

net = NeuralNetwork('models/100r100r2', device=device)

control = NeuralNetworkControl(net, device=device)
# controlif = NeuralNetworkControlIF(net, mode='ltv', method='CROWN', device=device)
controlif = NeuralNetworkControlIF(net, mode='hybrid', method='CROWN', device=device)
model = VehicleModel(control, controlif, u_step)
x0 = np.array([8,8,-2*np.pi/3,2])
pert = np.array([0.1,0.1,0.01,0.01])
# t_step = 0.01 if args.t_step is None else args.t_step
xlen = 4
print(torch.autograd.functional.jacobian(net, torch.Tensor(x0)))

t_step = u_step; p_step = u_step # rs plotting step
t_span = [0,args.t_end]
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


# eps, max_primer_depth, max_depth, cd, id, check_cont, dist, cut_dist
if EXPERIMENT == 1 :
    experiments = ((
                    ([0,0,0,0],0,0,1,1,0,0,False), 
                    ([0.2,0.2,np.inf,np.inf],1,2,0,0,0.1,0,False), 
                    ([0.25,0.25,np.inf,np.inf],1,2,0,0,0.1,0,False),
                   ), (
                    ([0,0,0,0],0,0,2,0,0,0,False), 
                    ([0.2,0.2,np.inf,np.inf],2,2,0,0,0.1,0,False), 
                    ([0.25,0.25,np.inf,np.inf],2,2,0,0,0.1,0,False),
                   ))
    fig1, axs1 = plt.subplots(2,3,dpi=100,figsize=[12,8],squeeze=False)
    fig1.subplots_adjust(left=0.05, right=0.95, bottom=0.075, top=0.95, wspace=0.125, hspace=0.25)
elif EXPERIMENT == 2 :
    experiments = ((
                    ([0,0,0,0],0,0,1,0,0,0,False), 
                    # ([0.2,0.2,np.inf,np.inf],1,2,0,0,0.1,0,False), 
                   ),)
    fig1, axs1 = plt.subplots(1,1,dpi=100,figsize=[5,5],squeeze=False)
    fig1.subplots_adjust(left=0.075, right=0.95, bottom=0.1, top=0.925, wspace=0.125, hspace=0.25)

# fig2, axs2 = plt.subplots(2,4,dpi=100,figsize=[14,8],squeeze=False)
# fig2.subplots_adjust(left=0.025, right=0.975, bottom=0.125, top=0.9, wspace=0.125, hspace=0.2)
table = [[r'$\varepsilon$',r'$D_p$',r'$D_\textsc{N}$',r'Runtime (s)',r'Volume']]

for i, exps in enumerate(experiments) :
    for j, exp in enumerate(exps) :
        # eps, max_primer_depth, max_depth, cd, id, check_contr = exp
        eps, max_primer_depth, max_depth, cd, id, check_contr, dist, cut_dist = exp
        # eps = eps*(pert/np.max(pert))
        # eps = eps*np.array([1,1,1,1])
        eps = np.asarray(eps)

        # rs = model.compute_reachable_set_eps(x0d, t_span, cd, id, 'euler', t_step, eps, max_depth, check_cont, False)
        # rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, enable_bar=False)
        runtimes = np.empty(runtime_N)
        for n in range(runtime_N) :
            if max_depth != 0 :
                rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, cut_dist, True, enable_bar=False)
                runtimes[n] = runtime
            else :
                # rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, cut_dist, True, enable_bar=True)
                rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, cd, id, 'euler', t_step, True, enable_bar=False)
                runtimes[n] = runtime
                # small hack for plotting purposes
                if n == runtime_N - 1 :
                    max_depth = cd + id
                    max_primer_depth = cd
        
        avg_runtime = np.mean(runtimes)
        std_runtime = np.std (runtimes)

        vol = volume(rs(t_span[1]))
        print(f'Runtime: {avg_runtime:.3f}$\pm${std_runtime:.3f}, Volume: {vol:.6f}\n')

        # axs1[i,j].text(-0.5,8.5,f'runtime:\n${runtime:.3f}$',fontsize=15,verticalalignment='top')

        axs1[i,j].text(-0.5,8.5,f'runtime: ${avg_runtime:.3f}\pm{std_runtime:.3f}$\nvolume: {vol:.5f}',fontsize=15,verticalalignment='top')
        # axs1[i,j].text(-0.5,8.5,f'runtime: $1.583 \pm 0.010$\nvolume: {vol:.5f}',fontsize=15,verticalalignment='top')

        delim = r'\,'
        table.append([rf'$[{delim.join(str(e) for e in eps)}]$'.replace('inf',r'\infty'),
                      rf'${max_depth}$', rf'${max_primer_depth}$', 
                      rf'${avg_runtime:.3f}\pm{std_runtime:.3f}$',rf'${vol:.3f}$'])

        for n in range(MC_N) :
            plot_Y_X(fig1, axs1[i,j], tt, trajs[n](tt)[0,:], trajs[n](tt)[1,:], xlim=[-1,9],ylim=[-1,9],lw=0.5,show_obs=(n==0))

        axs1[i,j].set_title(rf'$\varepsilon=${eps}, $D_p=${max_depth}, $D_\mathrm{{N}}=${max_primer_depth}'.replace('inf',r'$\infty$'))

        rs.draw_sg_boxes(axs1[i,j],np.arange(t_span[0],t_span[1]+p_step,p_step))
        # rs.draw_tree(axs2[i,j], prog='dot')
        # plt.ion()
        # plt.show()
        # input()

print(tabulate(table, tablefmt='latex_raw'))

fig1.savefig(rf'figures/cdc2023/veh_fig-exp{EXPERIMENT}.pdf')
plt.show()