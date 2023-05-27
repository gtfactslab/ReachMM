import numpy as np
import interval
import sympy as sp
from ReachMM import DiscreteTimeSpec, AutonomousSystem
from ReachMM import UniformPartitioner
import matplotlib.pyplot as plt

x1, x2 = sp.symbols('x1 x2')
f_eqns = [[
    x1**2 + 2*x1*x2 + x2**2,
    (x1 + x2)**2
],[
    4*sp.sin(x1/4)*sp.cos(x2/4) - 4*sp.cos(x1/4)*sp.sin(x2/4),
    4*sp.sin((x1-x2)/4)
]]

fig, axs = plt.subplots(1,2,dpi=100,figsize=[8,4],squeeze=False)
fig.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.95, wspace=0.15, hspace=0.25)

# f_inds = [(0,0,'r'),(0,1,'g'),(1,0,'b'),(1,1,'k')]
f_inds = [(0,0,'tab:green'),(1,1,'tab:blue')]

for f1, f2, color in f_inds :
    t_spec = DiscreteTimeSpec()
    sys = AutonomousSystem([x1, x2], [f_eqns[0][f1], f_eqns[1][f2]], t_spec)
    print(sys)
    partitioner = UniformPartitioner(sys)
    opts = UniformPartitioner.Opts(0,0)

    x0 = np.array([ np.interval(-1,1), np.interval(-1,1) ])
    rs = partitioner.compute_reachable_set(0,1,x0,opts)
    rs.draw_rs(axs[0,0], 1, color=color, label=f'[({f1}, {f2})]', lw=3)

for f1, f2, color in f_inds :
    t_spec = DiscreteTimeSpec()
    sys = AutonomousSystem([x1, x2], [f_eqns[0][f1], f_eqns[1][f2]], t_spec)
    print(sys)
    partitioner = UniformPartitioner(sys)
    opts = UniformPartitioner.Opts(5,0)

    x0 = np.array([ np.interval(-1,1), np.interval(-1,1) ])
    rs = partitioner.compute_reachable_set(0,1,x0,opts)
    rs.draw_rs(axs[0,1], 1, color=color, label=f'[({f1}, {f2})]', lw=3)

trajs = sys.compute_mc_trajectories(0, 1, x0, 2000)
mc_xx = np.array([traj(1) for traj in trajs])
axs[0,0].scatter(mc_xx[:,0], mc_xx[:,1], c='tab:red', s=2, zorder=0)
axs[0,1].scatter(mc_xx[:,0], mc_xx[:,1], c='tab:red', s=2, zorder=0)

xlim = [-2.25,4.25]; ylim = [-2.25,2.25]
axs[0,0].set_xlim(xlim); axs[0,1].set_xlim(xlim)
axs[0,0].set_ylim(ylim); axs[0,1].set_ylim(ylim)

# fig.legend()
fig.savefig('interval_example1.pdf')
plt.show()