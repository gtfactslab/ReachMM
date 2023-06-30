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
from inclusion import Ordering, all_corners
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

px, py, psi, v, u1, u2, w = sp.symbols('p_x p_y psi v u1 u2 w')

u2_lim = sp.pi/4
u2_softmax = lambda x : u2_lim*sp.tanh(x/u2_lim)
beta = sp.atan(sp.tan(u2)/2)
f_eqn = [
    v*sp.cos(psi + beta), 
    v*sp.sin(psi + beta), 
    v*sp.sin(beta),
    u1
]
uclip = np.array([
    np.interval(-20,20),
    np.interval(-np.pi/4,np.pi/4)
])

t_spec = ContinuousTimeSpec(0.125,0.125)
sys = System([px, py, psi, v], [u1, u2], [w], f_eqn, t_spec)
net = NeuralNetwork('models/100r100r2')
clsys = NNCSystem(sys, net, uclip=uclip)

print(sys.Df_x_sym)

cent = np.array([8,7,-2*np.pi/3,2])
pert = np.array([0.05,0.05,0.01,0.01])
x0 = from_cent_pert(cent, pert)

partitioner = UniformPartitioner(clsys)
part_opts = UniformPartitioner.Opts(0,0)
t_span = [0,1.5]
tt = t_spec.tt(*t_span)

# Monte Carlo Plotting
MC_N = 100
trajs = clsys.compute_mc_trajectories(*t_span, x0, MC_N)

def run_and_plot (ax, title) :
    print('\n',title)
    rs, times = run_times(args.runtime_N, partitioner.compute_reachable_set, *t_span, x0, part_opts)
    avg_runtime = np.mean(times); std_runtime = np.std(times)
    print(f'{avg_runtime:.3f} ± {std_runtime:.3f}')

    rs.draw_rs(ax, tt, color='tab:blue')

    ax.add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon',zorder=0))

    ax.set_xlim([-0.5,8.5])
    ax.set_ylim([-0.5,8.5])
    ax.set_xlabel('$p_x$',labelpad=3); ax.set_ylabel('$p_y$',labelpad=3, rotation='horizontal')

    ax.text(0,8,f'runtime: ${avg_runtime:.3f}\pm{std_runtime:.3f}$',fontsize=15,verticalalignment='top')
    ax.set_title(title)

    for traj in trajs :
        traj.plot2d(ax, tt, c='tab:red', zorder=0)

# fig, axs = plt.subplots(2,2,dpi=100,figsize=[8,8],squeeze=False)
# fig.subplots_adjust(left=0.075, right=0.95, bottom=0.125, top=0.925, wspace=0.125, hspace=0.25)
fig, axs = plt.subplots(2,3,dpi=100,figsize=[12,8],squeeze=False)
fig.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.925, wspace=0.125, hspace=0.25)

# Global Interconnection Mode
clsys.incl_opts = NNCSystem.InclOpts('interconnect')
clsys.control.mode = 'global'
run_and_plot(axs[0,0], f'Input-Output Mode: $\\mathsf{{F}}^\\mathrm{{io}}$')
clsys.control.mode = 'hybrid'

# Hybrid Interconnection Mode
clsys.incl_opts = NNCSystem.InclOpts('interconnect')
run_and_plot(axs[1,0], f'Interconnection: $\\mathsf{{F}}^\\mathrm{{int}}$')

# Cornered Mode
ax = axs[0,1]
clsys.incl_opts = NNCSystem.InclOpts('jacobian')
clsys.set_two_corners()
run_and_plot(axs[0,1], f'Cornered: $\\mathsf{{F}}^\\mathrm{{jac}}$')

# Cornered+Interconnection Mode
clsys.incl_opts = NNCSystem.InclOpts('jacobian+interconnect')
clsys.set_two_corners()
run_and_plot(axs[1,1], f'Interconnection + Cornered: $\\mathsf{{F}}^\\mathrm{{int}}\\wedge \\mathsf{{F}}^\\mathrm{{jac}}$')

# Mixed States Cornered Mode
clsys.incl_opts = NNCSystem.InclOpts('jacobian', orderings=[Ordering((0,1,2,3))])
clsys.set_two_corners()
run_and_plot(axs[0,2], f'Mixed-States Cornered Jacobian: $\\mathsf{{F}}^\\mathrm{{jac}}$')

# Mixed States and Control Cornered Mode
clsys.incl_opts = NNCSystem.InclOpts('jacobian', orderings=[Ordering((0,1,2,3,4,5))])
clsys.set_two_corners()
clsys.set_standard_ordering()
run_and_plot(axs[1,2], f'Mixed-All Cornered Jacobian: $\\mathsf{{F}}^\\mathrm{{jac}}$')


plt.savefig('figures/veh_tac2023.pdf')
plt.show()

