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


def platoon (N, show_plot, rset=1, kp=5, kv=5, u_lim=4.0) :
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

    t_spec = ContinuousTimeSpec(0.025,0.25)
    sys = System(x_vars, u_vars, w_vars, f_eqn, t_spec)
    print(sys.Df_x_sym)
    # print(sys.Df_u_sym)

    net = NeuralNetwork('models/100r100r2')
    net.seq.insert(0,nn.Linear(len(x_vars), 4))
    W = torch.zeros((4,len(x_vars)))
    W[0,0] = 1; W[1,1] = 1; W[2,2] = 1; W[3,3] = 1
    net.seq[0].weight = nn.Parameter(W)

    dist = UniformDisturbance([np.interval(-0.005,0.005) for w in w_vars])
    clsys = NNCSystem(sys, net, 'jacobian', dist=dist)
                    #   g_tuple=(x_vars,), g_eqn=list(x[0]))
    t_end = 3
    print(clsys)
    partitioner = UniformPartitioner(clsys)
    popts = UniformPartitioner.Opts(0,0)

    # x0_1 = [7.0,7.0,2.0,-4.0]
    x0 = []
    ang = 3*np.pi/4
    for i in range(N) :
        x0.extend([7.0+rset*i*np.cos(ang),7.0+rset*i*np.sin(ang),2.0,-4.0])
    x0 = np.asarray(x0)
    print(x0)

    def run () :
        # return clsys.compute_trajectory(0,t_end,x0.astype(np.interval))
        return partitioner.compute_reachable_set(0,t_end,x0.astype(np.interval))
    rs, runtimes = run_times(1, run)
    print(rs(t_end))
    print(f'Runtime: {np.mean(runtimes)} \\pm {np.std(runtimes)}')

    if not show_plot :
        return runtimes

    tt = t_spec.tt(0,t_end)
    xx = rs(tt)

    fig1, axs1 = plt.subplots(1,1,figsize=[4,6],squeeze=False)
    fig1.subplots_adjust(left=0.075, right=0.95, bottom=0.125, top=0.925, wspace=0.125, hspace=0.25)

    fig2, axs2 = plt.subplots(2,2,figsize=[8,8],squeeze=False)
    fig2.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)

    colors = list(mcolors.TABLEAU_COLORS)

    for i in range(N) :
        xx_i = xx[:,4*i:4*(i+1)]
        plot_iarray_t(axs2[0,0], tt, xx_i[:,0], color=colors[i], alpha=0.25, label=f'px_{i}')
        plot_iarray_t(axs2[0,1], tt, xx_i[:,1], color=colors[i], alpha=0.25, label=f'py_{i}')
        plot_iarray_t(axs2[1,0], tt, xx_i[:,2], color=colors[i], alpha=0.25, label=f'vx_{i}')
        plot_iarray_t(axs2[1,1], tt, xx_i[:,3], color=colors[i], alpha=0.25, label=f'vy_{i}')
        for ax in axs2.reshape(-1):
            ax.legend()

    axs1[0,0].add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon',zorder=0))


    dim = np.ceil(np.sqrt(N)).astype(int)

    fig3, axs3 = plt.subplots(dim,dim,figsize=[8,8],squeeze=False)
    fig3.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)

    MC_N = 100
    for mc_i in range(MC_N) :
        mc_traj = clsys.compute_trajectory(0,t_end,x0)
        mc_xx = mc_traj(tt)
        for i in range(N) :
            mc_xx_i = mc_xx[:,4*i:4*(i+1)]
            axs1[0,0].plot(mc_xx_i[:,0], mc_xx_i[:,1], zorder=0, color=colors[i])
            axs3.reshape(-1)[i].scatter(mc_xx_i[:,0], mc_xx_i[:,1], zorder=0, s=0.1, color='tab:red')
    
    for axi, ax in enumerate(axs3.reshape(-1)) :
        for i in range(N) :
            if i == axi :
                rs.draw_rs(ax,tt,4*i,4*i+1)
            else :
                pass
        ax.set_xlim([-1,10])
        ax.set_ylim([-1,10])

    plt.show()
    return runtimes

if __name__ == '__main__' :
    platoon(9, True)
