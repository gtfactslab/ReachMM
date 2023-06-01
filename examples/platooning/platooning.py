import numpy as np
import interval
from interval import from_cent_pert, get_lu, get_cent_pert
import torch
import torch.nn as nn
import sympy as sp
from ReachMM.time import *
from ReachMM.system import *
from ReachMM.reach import UniformPartitioner, CGPartitioner
from ReachMM.control import ConstantDisturbance, UniformDisturbance
from ReachMM.utils import run_times, draw_iarray, plot_iarray_t
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from inspect import getsource


def platoon (N, show_plot, runtime_N, rset=0.5, kp=5, kv=5, u_lim=5.0) :
    x_vars = []
    u_vars = []
    w_vars = []
    f_eqn = []
    x = []

    u_softmax = lambda x : u_lim*sp.tanh(x/u_lim)

    for i in range (N) :
        px_i, py_i, vx_i, vy_i = (x_i := sp.symbols(f'px_{i} py_{i} vx_{i} vy_{i}'))
        x.append(x_i)
        x_vars.extend(list(x_i))
    
    for i in range (N) :
        px_i, py_i, vx_i, vy_i = x[i]
        wx_i, wy_i = (w_i := sp.symbols(f'wx_{i} wy_{i}'))
        if i == 0 :
            # Only first unit is controlled by NN
            ux_i, uy_i = (u := sp.symbols('u1 u2'))
            u_vars.extend(list(u))
        else :
            if i < N-1:
                # Middle units communicate with nearest two
                # neighbors = [i-1, i+1]
                neighbors = [i-1]
            elif i == N-1 :
                # Final unit communicates with nearest one
                neighbors = [i-1]
            ux_i = 0
            uy_i = 0
            for j in neighbors :
                vmag_j = sp.sqrt(x[j][2]**2 + x[j][3]**2)
                px_des = x[j][0] - rset*(x[j][2] / vmag_j)
                py_des = x[j][1] - rset*(x[j][3] / vmag_j)
                vx_des = x[j][2]
                vy_des = x[j][3]
                ux_i += kp*(px_des - x[i][0]) + kv*(vx_des - x[i][2])
                uy_i += kp*(py_des - x[i][1]) + kv*(vy_des - x[i][3])
        
        f_i = [
            vx_i,
            vy_i,
            u_softmax(ux_i) + wx_i,
            u_softmax(uy_i) + wy_i
        ]
        
        w_vars.extend(list(w_i))
        f_eqn.extend(f_i)
    
    print("Number of states: ", len(x_vars))
    # print(u_vars)
    # print(w_vars)
    # print(f_eqn)

    t_spec = ContinuousTimeSpec(0.0125,0.0125)
    sys = System(x_vars, u_vars, w_vars, f_eqn, t_spec)
    # print(sys.Df_x_sym)
    # print(sys.Df_u_sym)

    net = NeuralNetwork('models/100r100r2')
    # net.seq.insert(0,nn.Linear(len(x_vars), 4))
    # W = torch.zeros((4,len(x_vars)))
    # W[0,0] = 1; W[1,1] = 1; W[2,2] = 1; W[3,3] = 1
    # net.seq[0].weight = nn.Parameter(W)

    dist = UniformDisturbance([np.interval(-0.001,0.001) for w in w_vars])
    clsys = NNCSystem(sys, net, 'interconnect', dist=dist,
                      g_tuple=(x_vars,), g_eqn=list(x[0]))
    t_end = 1.5
    print(clsys)
    partitioner = UniformPartitioner(clsys)
    popts = UniformPartitioner.Opts(0,0)

    cent = [7.25,5.75,-0.5,-5]
    # cent = [7,5.75,-0.5,-4.5]
    # cent = [8,8,-4.0,-4.0]
    pert = [0.025,0.025,0,0]
    x0cent = []
    x0pert = []
    # ang = 7*np.pi/12
    ang = (0.5/1.5)*np.pi
    for i in range(N) :
        x0cent.extend([cent[0]+rset*i*np.cos(ang),cent[1]+rset*i*np.sin(ang),cent[2],cent[3]])
        x0pert.extend(pert)
    x0 = from_cent_pert(x0cent, x0pert)
    # print(x0)

    def run () :
        # return clsys.compute_trajectory(0,t_end,x0.astype(np.interval))
        return partitioner.compute_reachable_set(0,t_end,x0.astype(np.interval))
    rs, runtimes = run_times(runtime_N, run)
    # print(rs(t_end))
    print(f'Runtime: {np.mean(runtimes)} \\pm {np.std(runtimes)}')

    if not show_plot :
        return runtimes

    tt = t_spec.tt(0,t_end)
    xx = rs(tt)

    plt.rc('font', size=14)

    fig1, axs1 = plt.subplots(1,1,figsize=[4,6],squeeze=False)
    fig1.subplots_adjust(left=0.075, right=0.95, bottom=0.125, top=0.925, wspace=0.125, hspace=0.25)

    fig2, axs2 = plt.subplots(4,1,figsize=[3,8],dpi=100,squeeze=False,sharex='col')
    fig2.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.95, wspace=0.2, hspace=0.05)

    colors = list(mcolors.TABLEAU_COLORS)

    for i in range(N) :
        xx_i = xx[:,4*i:4*(i+1)]
        state = ['p_x', 'p_y', 'v_x', 'v_y']
        for ax_i, ax in enumerate(axs2.reshape(-1)) :
            ax.locator_params('x', min_n_ticks=4, nbins=4)
            ax.locator_params('y', min_n_ticks=4, nbins=4)
            plot_iarray_t(ax, tt, xx_i[:,ax_i], color=colors[i], alpha=0.25, label=f'{state[ax_i]}_{i}')
            ax.set_xlabel('time (s)', labelpad=0.1)
            ax.text(0.1,0.4,f'${state[ax_i]}$',fontsize=15,verticalalignment='top',transform=ax.transAxes)
            # ax.set_ylabel(f'${state[ax_i]}$', labelpad=0.1, rotation='horizontal')
            # ax.legend(loc='lower left')

    axs1[0,0].add_patch(Circle((4,4),3/1.33,lw=0,fc='salmon',zorder=0))


    dim = np.ceil(np.sqrt(N)).astype(int)

    # fig3, axs3 = plt.subplots(dim,dim,figsize=[8,8],squeeze=False,sharex='col',sharey='row')
    fig3, axs3 = plt.subplots(dim,dim,figsize=[8,8],dpi=100,squeeze=False,sharex='all',sharey='all')
    fig3.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.05, hspace=0.05)

    MC_N = 100
    mc_trajs = clsys.compute_mc_trajectories(0,t_end,x0,MC_N)
    for mc_traj in mc_trajs :
        mc_xx = mc_traj(tt)
        for i in range(N) :
            mc_xx_i = mc_xx[:,4*i:4*(i+1)]
            axs1[0,0].plot(mc_xx_i[:,0], mc_xx_i[:,1], zorder=0, color=colors[i])
            # axs3.reshape(-1)[i].scatter(mc_xx_i[:,0], mc_xx_i[:,1], zorder=0, s=0.1, color='tab:red')
    
    for axi, ax in enumerate(axs3.reshape(-1)) :
        ax.add_patch(Circle((4,4),3/1.33,lw=0,fc='salmon',zorder=0))
        for i in range(N) :
            if i == axi :
                rs.draw_rs(ax,tt,4*i,4*i+1,fc='tab:blue',ec='none',alpha=1)
                ax.add_patch(Circle((x0cent[4*i], x0cent[4*i + 1]), 0.25, lw=0, fc=colors[i]))
            else :
                ax.add_patch(Circle((x0cent[4*i], x0cent[4*i + 1]), 0.15, alpha=0.25, lw=0, fc='grey'))
        ax.set_xlim([-1,10])
        ax.set_ylim([-1,10])
        ax.xaxis.set_major_locator(plt.MultipleLocator(base=2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(base=2))
        # ax.set_xticks([0,2,4,6,8,10])
        # ax.set_yticks([0,2,4,6,8,10])

    fig1.savefig('figures/platooning_fig1.pdf')
    fig2.savefig('figures/platooning_fig2.pdf')
    fig3.savefig('figures/platooning_fig3.pdf')
    plt.show()
    return runtimes

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(description="Platooning Experiments")
    parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",
                        type=int, default=1)
    args = parser.parse_args()

    from tabulate import tabulate

    table = [['$N$ (units)', 'States', 'Runtime (s)']]
    for N in [1,4,9,20,50] :
        runtimes = platoon(N, False, args.runtime_N)
        table.append([N, 4*N, rf'${np.mean(runtimes)} \pm {np.std(runtimes)}$'])

    print(tabulate(table,tablefmt="latex_raw"))

    platoon(9, True, 1)
