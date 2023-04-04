import argparse

parser = argparse.ArgumentParser(description="Double Integrator Experiments for CDC Paper")

parser.add_argument('-T', '--t_end', help="End time for simulation",\
                    type=int, default=6)
parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",\
                    type=int, default=1)
parser.add_argument('--tree', help="Enable for Tree (Figure 2)", \
                    default=False, action='store_true')

args = parser.parse_args()

from ReachMM import NeuralNetworkControl, NeuralNetworkControlIF
from ReachMM import NeuralNetwork
from ReachMM.utils import run_time, gen_ics_pert
from ReachMM.reach import volume
from DoubleIntegrator import *
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tabulate import tabulate
import shapely.geometry as sg
import shapely.ops as so
import polytope

runtime_N = args.runtime_N

EXPERIMENT = 1 if args.tree is False else 2

# device = 'cuda:0'
device = 'cpu'
netpath = '10r5r1'
MC_N = 200

t_step = 1
t_span = [0,args.t_end]
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

if EXPERIMENT == 1:
    experiments = ((
        (0,0,0,2,0,0.5,0,False), 
        (0,0,0,2,4,0.5,0,False)
    ), (
        (0.1,1,3,0,0,0,0,False), 
        (0.05,2,6,0,0,0,0,False)
    ))
    fig1, axs1 = plt.subplots(2,2,dpi=100,figsize=[8,8],squeeze=False)
    fig1.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.95, wspace=0.15, hspace=0.25)
elif EXPERIMENT == 2:
    experiments = ((
        (0.1,2,10,0,0,0,0,False),
    ),)
    fig1, axs1 = plt.subplots(1,1,dpi=100,figsize=[8,8],squeeze=False)
    fig1.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.95, wspace=0.15, hspace=0.25)

    fig2, axs2 = plt.subplots(2,3,dpi=100,figsize=[14,8],squeeze=False)
    fig2.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0, hspace=0.075)

table = [[r'Method',r'Setup',r'Runtime (s)',r'Area']]

rss = []

for i, exps in enumerate(experiments) :
    rssi = []
    for j, exp in enumerate(exps) :
        eps, max_primer_depth, max_depth, cd, id, check_contr, dist, cut_dist = exp

        runtimes = np.empty(runtime_N)
        for n in range(runtime_N) :
            if max_depth != 0 :
                rs, runtime = run_time(model.compute_reachable_set_eps, x0d, 
                                       t_span, cd, id, 'euler', t_step, eps, 
                                       max_primer_depth, max_depth, check_contr, 
                                       cut_dist, False, enable_bar=False, 
                                       axs=None if EXPERIMENT==1 else axs2.reshape(-1))
                runtimes[n] = runtime
                cg = '-CG'
            else :
                # rs, runtime = run_time(model.compute_reachable_set_eps, x0d, t_span, cd, id, 'euler', t_step, eps, max_primer_depth, max_depth, check_contr, cut_dist, True, enable_bar=True)
                rs, runtime = run_time(model.compute_reachable_set, x0d, 
                                       t_span, cd, id, 'euler', t_step, 
                                       False, enable_bar=False)
                runtimes[n] = runtime
                # small hack for plotting purposes
                if n == runtime_N - 1 :
                    max_depth = cd + id
                    max_primer_depth = cd
                    cg = ''
        
        avg_runtime = np.mean(runtimes)
        std_runtime = np.std (runtimes)

        axs1[i,j].set_xlim([-1,3.5])
        axs1[i,j].set_ylim([-1.5,1])

        # vol = volume(rs(t_span[1]))
        vol = rs.area(t_span[1])
        print(f'Runtime: {avg_runtime:.3f}$\pm${std_runtime:.3f}, Area: {vol:.6f}\n')

        axs1[i,j].text(-0.75,0.85,f'runtime: ${avg_runtime:.3f}\pm{std_runtime:.3f}$\narea: {vol:.5f}',fontsize=15,verticalalignment='top')

        table.append([r'ReachMM'+cg, rf'${eps}$, ${max_depth}$, ${max_primer_depth}$', 
                      rf'${avg_runtime:.3f}\pm{std_runtime:.3f}$',rf'${vol:.1E}}}$'.replace('E-0',r'\times10^{-')])

        for n in range(MC_N) :
            axs1[i,j].scatter(trajs[n](tt)[0,:],trajs[n](tt)[1,:], s=0.25, c='r')

        rs.draw_sg_boxes(axs1[i,j],tt)
        axs1[i,j].set_title(rf'$\varepsilon=${eps}, $D_p=${max_depth}, $D_\mathrm{{N}}=${max_primer_depth}',fontsize=15)
        # rs.draw_tree(axs2[i,j], prog='dot')
        # plt.ion()
        # plt.show()
        # input()
        rssi.append(rs)
    rss.append(rssi)

print(rss)

if EXPERIMENT == 1 :
    fig3, axs3 = plt.subplots(1,2,dpi=100,figsize=[11,5],squeeze=True)
    fig3.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.125, hspace=0.2)

    LP_parts = ['4.npy', '16.npy']
    # LP_parts = ['55.npy', '205.npy']
    LipBnB_eps = ['0.1.npy', '0.001.npy']

    for axi, ax in enumerate(axs3) :
        ax.set_xlim([-1,3.5])
        ax.set_ylim([-1.5,1])

        for n in range(MC_N) :
            ax.scatter(trajs[n](tt)[0,:],trajs[n](tt)[1,:], s=0.25, c='r')

        # ReachMM-CG
        rss[1][axi].draw_sg_boxes(ax,tt)

        # ReachLP-Uniform
        # LP = np.load('comparisons/ReachLP-Results/GreedySimGuided-' + LP_parts[axi], allow_pickle=True)
        LP = np.load('comparisons/ReachLP-Results/Uniform-' + LP_parts[axi], allow_pickle=True)
        LP_rs = np.array([a[1] for a in LP])
        print('ReachLP Areas for Uniform-' + LP_parts[axi])
        for t in tt[1:6] :
            boxes = [sg.box(box[0,0],box[1,0],box[0,1],box[1,1]) for box in LP_rs[:,t-1,:,:]]
            shape = so.unary_union(boxes)
            xs, ys = shape.exterior.xy    
            ax.fill(xs, ys, alpha=1, fc='none', ec='tab:orange')
            print(shape.area)

        # ReachLipBnB
        BnB_AAs = np.load('comparisons/ReachLipBnB-Results/AAs-' + LipBnB_eps[axi])
        BnB_bbs = np.load('comparisons/ReachLipBnB-Results/bbs-' + LipBnB_eps[axi])
        print('ReachLipBnB Areas for AAs/bbs-' + LipBnB_eps[axi])
        for k in range(len(BnB_AAs)) :
            AA = BnB_AAs[k]; bb = BnB_bbs[k]
            pltp = polytope.Polytope(AA, bb)
            lipbnb = pltp.plot(ax, alpha=1, color='none', edgecolor='tab:green', linewidth=1, linestyle='-')
            lipbnb.set_label('ReachLipBnB')
            print(pltp.volume)

        legendhack = [
            Line2D([0], [0], lw=1, color='tab:blue', label='ReachMM-CG'),
            Line2D([0], [0], lw=1, color='tab:orange', label='ReachLP-Uniform'),
            Line2D([0], [0], lw=1, color='tab:green', label='ReachLipBnB'),
        ]

        ax.legend(legendhack, ['ReachMM-CG', 'ReachLP-Uniform', 'ReachLipBnB'])

    print(tabulate(table, tablefmt='latex_raw'))

    fig3.savefig(r'figures/cdc2023/DI_fig3.pdf')
else :
    fig2.savefig(r'figures/cdc2023/DI_fig2.pdf')

fig1.savefig(r'figures/cdc2023/DI_fig1.pdf')
plt.show()