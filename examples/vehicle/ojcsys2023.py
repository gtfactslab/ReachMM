import argparse
parser = argparse.ArgumentParser(description="Vehicle (bicycle model) Experiments for L4DC 2023 Paper")
parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",\
                    type=int, default=1)
args = parser.parse_args()

import numpy as np
import interval
from interval import from_cent_pert
import sympy as sp

from ReachMM import ContinuousTimeSpec, DiscretizedTimeSpec
from ReachMM import System, NeuralNetwork, NNCSystem
from ReachMM import UniformPartitioner, CGPartitioner
from ReachMM.utils import run_times
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

px, py, psi, v, u1, u2, w = sp.symbols('p_x p_y psi v u1 u2 w')

beta = sp.atan(sp.tan(u2)/2)
f_eqn = [
    v*sp.cos(psi + beta), 
    v*sp.sin(psi + beta), 
    v*sp.sin(beta),
    u1
]
x_clip = np.array([
    np.interval(-np.inf, np.inf),
    np.interval(-np.inf, np.inf),
    np.interval(-np.inf, np.inf),
    # np.interval(-np.pi, np.pi),
    np.interval(-np.inf, np.inf),
])
uclip = np.array([
    np.interval(-20,20),
    np.interval(-np.pi/4,np.pi/4)
])

t_spec = ContinuousTimeSpec(0.125,0.125)
# t_spec = DiscretizedTimeSpec(0.125)
sys = System([px, py, psi, v], [u1, u2], [w], f_eqn, t_spec, x_clip)
net = NeuralNetwork('models/100r100r2')
clsys = NNCSystem(sys, net, 'interconnect')
# clsys = NNCSystem(sys, net, 'jacobian')

t_span = [0,1.5]

# cent = np.array([8,8,-2*np.pi/3,2])
# pert = np.array([0.05,0.05,0.005,0.005])
cent = np.array([8,7,-2*np.pi/3,2])
pert = np.array([0.05,0.05,0.01,0.01])
# pert = np.array([0.05,0.05,0.005,0.005])
# cent = np.array([8,8,-2*np.pi/3,2])
# pert = np.array([0.1,0.1,0.01,0.01])
x0 = from_cent_pert(cent, pert)

partitioner = UniformPartitioner(clsys)
# partitioner = CGPartitioner(clsys)

# Experiment 1
# opts = CGPartitioner.Opts(0.25, 0.1, 2, 0)
opts = UniformPartitioner.Opts(1,0)
rs, times = run_times(args.runtime_N, partitioner.compute_reachable_set, t_span[0], t_span[1], x0, opts)
avg_runtime = np.mean(times); std_runtime = np.std(times)

print(avg_runtime, '\pm', std_runtime)


fig, axs = plt.subplots(1,1,dpi=100,figsize=[4,4],squeeze=False)
# fig.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.95, wspace=0.15, hspace=0.25)
fig.subplots_adjust(left=0.075, right=0.95, bottom=0.125, top=0.925, wspace=0.125, hspace=0.25)


tt = t_spec.tt(t_span[0], t_span[1])
rs.draw_rs(axs[0,0], tt, color='tab:blue')

MC_N = 100
trajs = clsys.compute_mc_trajectories(*t_span, x0, MC_N)
for traj in trajs :
    traj.plot2d(axs[0,0], tt, c='tab:red', zorder=0)

axs[0,0].add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon',zorder=0))

axs[0,0].set_xlim([-0.5,8.5])
axs[0,0].set_ylim([-0.5,8.5])
axs[0,0].set_xlabel('$p_x$',labelpad=3); axs[0,0].set_ylabel('$p_y$',labelpad=3, rotation='horizontal')

axs[0,0].text(0,8,f'runtime: ${avg_runtime:.3f}\pm{std_runtime:.3f}$',fontsize=15,verticalalignment='top')

print(rs(t_span[1]))
# fig.savefig('figures/wfvml2023.pdf')
plt.show()

