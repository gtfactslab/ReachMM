from ReachMM import NeuralNetworkControl, NeuralNetworkControlIF
from ReachMM import NeuralNetwork
from ReachMM.utils import run_time, gen_ics_pert
from ReachMM.reach import volume
from DoubleIntegrator import *
import torch
import matplotlib.pyplot as plt
import numpy as np

runtime_N = 1

# device = 'cuda:0'
device = 'cpu'
netpath = '10r5r1-LowLip'
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

X0 = gen_ics_pert(x0, pert, MC_N)
trajs = []
for mc_x0 in X0 :
    trajs.append(model.compute_trajectory(mc_x0, t_span, 'euler', t_step, enable_bar=False))

# eps, max_primer_depth, max_depth, cd, id, check_cont, dist, cut_dist
# experiments = (((0,0,0,0,0,0.5,0,False), (  0,0,0,2,0,0.5,0,False), (  1,0,0,2,2,0.5,0,False), (   0,0,0,3,2,0.5,0,False)),
#                ((0.1,1,3,0,0,0,0,False), (0.1,2,4,0,0,0,0,False), (0.1,2,10,0,0,0,0,False), (0.07,2,10,0,0,0,0,False)))
experiments = (((0,0,0,2,0,0.5,0,False), (0,0,0,2,4,0.5,0,False)),
               ((0.1,1,3,0,0,0,0,False), (0.05,2,6,0,0,0,0,False)))

# fig1, axs1 = plt.subplots(2,4,dpi=100,figsize=[14,8],squeeze=False)
# fig1.subplots_adjust(left=0.025, right=0.975, bottom=0.125, top=0.9, wspace=0.125, hspace=0.2)
fig1, axs1 = plt.subplots(2,2,dpi=100,figsize=[8,8],squeeze=False)
fig1.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.95, wspace=0.15, hspace=0.25)
# fig2, axs2 = plt.subplots(2,4,dpi=100,figsize=[14,8],squeeze=False)
# fig2.subplots_adjust(left=0.025, right=0.975, bottom=0.125, top=0.9, wspace=0.125, hspace=0.2)

for i, exps in enumerate(experiments) :
    for j, exp in enumerate(exps) :
        eps, max_primer_depth, max_depth, cd, id, check_contr, dist, cut_dist = exp

        runtimes = np.empty(runtime_N)
        for n in range(runtime_N) :
            if max_depth != 0 :
                rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, cut_dist, False, enable_bar=False)
                runtimes[n] = runtime
            else :
                # rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, cut_dist, True, enable_bar=True)
                rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, cd, id, 'euler', t_step, False, enable_bar=False)
                runtimes[n] = runtime
                # small hack for plotting purposes
                if n == runtime_N - 1 :
                    max_depth = cd + id
                    max_primer_depth = cd
        
        avg_runtime = np.mean(runtimes)
        std_runtime = np.std (runtimes)

        axs1[i,j].set_xlim([-1,3.5])
        axs1[i,j].set_ylim([-1.5,1])

        vol = volume(rs(t_span[1]))
        print(f'Runtime: {avg_runtime:.3f}$\pm${std_runtime:.3f}, Volume: {vol:.6f}\n')

        axs1[i,j].text(-0.75,0.85,f'runtime: ${avg_runtime:.3f}\pm{std_runtime:.3f}$\nvolume: {vol:.5f}',fontsize=15,verticalalignment='top')


        for n in range(MC_N) :
            axs1[i,j].scatter(trajs[n](tt)[0,:],trajs[n](tt)[1,:], s=0.25, c='r')

        rs.draw_sg_boxes(axs1[i,j],tt)
        axs1[i,j].set_title(rf'$\varepsilon=${eps}, $D_p=${max_depth}, $D_\mathrm{{N}}=${max_primer_depth}',fontsize=15)
        # rs.draw_tree(axs2[i,j], prog='dot')
        # input()

fig1.savefig(r'figures/cdc2023/DI_fig1.pdf')
plt.show()