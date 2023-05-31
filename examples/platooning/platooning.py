import numpy as np
import interval
from interval import from_cent_pert, get_lu, get_cent_pert
import torch
import torch.nn as nn
import sympy as sp
from ReachMM.time import *
from ReachMM.system import *
from ReachMM.reach import UniformPartitioner, CGPartitioner
from ReachMM.control import ConstantDisturbance
from ReachMM.utils import run_times, draw_iarray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from inspect import getsource

px, py, vx, vy, ux, uy, w = sp.symbols('px py vx vy ux uy w')
x_vars = [px, py, vx, vy]
u_vars = [ux, uy]
w_vars = [w]

u_lim = 4.0
u_softmax = lambda x : u_lim*sp.tanh(x/u_lim)

f_eqn = [
    vx,
    vy,
    # ux,
    # uy
    u_lim*sp.tanh(ux/u_lim),
    u_lim*sp.tanh(uy/u_lim) 
]

t_spec = ContinuousTimeSpec(0.025,0.25)
# t_spec = DiscretizedTimeSpec(0.25)
sys = System(x_vars, u_vars, w_vars, f_eqn, t_spec)
net = NeuralNetwork('models/100r100r2')

print(sys.Df_u_sym)

# extra1 = nn.Linear(2,2,False)
# extra1.weight = nn.Parameter(torch.tensor([ [1/u_lim, 0], [0, 1/u_lim] ]))
# extra2 = nn.Linear(2,2,False)
# extra2.weight = nn.Parameter(torch.tensor([ [u_lim, 0], [0, u_lim] ]))
# net.seq.append(extra1)
# net.seq.append(nn.Tanh())
# net.seq.append(extra2)

clsys = NNCSystem(sys, net, 'jacobian')
t_end = 3
print(clsys)

partitioner = UniformPartitioner(clsys)
popts = UniformPartitioner.Opts(0,0)
# partitioner = CGPartitioner(clsys)
# popts = CGPartitioner.Opts(0.05, 0.2, 2, 0)

x0 = np.array([
    np.interval(6.9,7.1),
    np.interval(6.9,7.1),
    np.interval(1.99,2.01),
    np.interval(-4.01,-3.99)
])

fig, axs = plt.subplots(2,2,dpi=100,figsize=[8,8],squeeze=False)
# fig.subplots_adjust(left=0.075, right=0.95, bottom=0.125, top=0.925, wspace=0.125, hspace=0.25)

tt = t_spec.tt(0,t_end)

def run () :
    rs = partitioner.compute_reachable_set(0,t_end,x0,popts)
    safe = 'T' # safe = rs.check_safety(spec_lam, tt)
    return rs, safe
(rs, safe), times = run_times(1, run)

print(f'Safe: {safe} in {np.mean(times)} \\pm {np.std(times)} (s)')

xx = rs(tt)
print(rs(t_end))
rs.draw_rs(axs[0,0], tt[::5])


trajs = clsys.compute_mc_trajectories(0,t_end,x0,100)
tt = clsys.sys.t_spec.tt(0,t_end)
for traj in trajs :
    axs[0,0].plot(traj(tt)[:,0], traj(tt)[:,1], color='tab:red', zorder=0)
    axs[1,0].plot(tt, traj(tt)[:,0], color='tab:red')
    axs[1,0].plot(tt, traj(tt)[:,1], color='tab:red')
    axs[1,1].plot(tt, traj(tt)[:,2], color='tab:red')
    axs[1,1].plot(tt, traj(tt)[:,3], color='tab:red')

axs[0,0].add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon',zorder=0))

axs[0,0].set_xlim([-1,8.5])
axs[0,0].set_ylim([-1,8.5])
axs[0,0].set_xlabel('$p_x$',labelpad=3); axs[0,0].set_ylabel('$p_y$',labelpad=3, rotation='horizontal')

axs[0,1].plot(tt,np.norm(rs(tt)[:,0]),color='tab:blue',label='pxwidth')
axs[0,1].plot(tt,np.norm(rs(tt)[:,1]),color='tab:green',label='pywidth')
axs[0,1].plot(tt,np.norm(rs(tt)[:,2]),label='vxwidth')
axs[0,1].plot(tt,np.norm(rs(tt)[:,3]),label='vywidth')
axs[0,1].legend()

pxl, pxu = get_lu(xx[:,0])
pyl, pyu = get_lu(xx[:,1])
axs[1,0].plot(tt, pxl, color='tab:blue')
axs[1,0].plot(tt, pxu, color='tab:blue')
axs[1,0].fill_between(tt,pxl,pxu,color='tab:blue',alpha=0.25,label='$p_x$')
axs[1,0].plot(tt, pyl, color='tab:green')
axs[1,0].plot(tt, pyu, color='tab:green')
axs[1,0].fill_between(tt,pyl,pyu,color='tab:green',alpha=0.25,label='$p_y$')
axs[1,0].legend()

vxl, vxu = get_lu(xx[:,2])
vyl, vyu = get_lu(xx[:,3])
# vxl, vxu = get_lu(rs.subpartitions[1](tt)[:,2])
# vyl, vyu = get_lu(rs.subpartitions[1](tt)[:,3])
axs[1,1].plot(tt, vxl, color='tab:blue')
axs[1,1].plot(tt, vxu, color='tab:blue')
axs[1,1].fill_between(tt,vxl,vxu,color='tab:blue',alpha=0.25,label='$v_x$')
axs[1,1].plot(tt, vyl, color='tab:green')
axs[1,1].plot(tt, vyu, color='tab:green')
axs[1,1].fill_between(tt,vyl,vyu,color='tab:green',alpha=0.25,label='$v_y$')
axs[1,1].legend()


plt.show()