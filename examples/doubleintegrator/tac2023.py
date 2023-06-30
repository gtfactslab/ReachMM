import argparse
parser = argparse.ArgumentParser(description="Vehicle (bicycle model) Experiments for L4DC 2023 Paper")
parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",\
                    type=int, default=1)
args = parser.parse_args()

import numpy as np
import interval
from interval import from_cent_pert
import sympy as sp

from ReachMM import DiscreteTimeSpec, ContinuousTimeSpec
from ReachMM import System, NeuralNetwork, NNCSystem
from ReachMM import UniformPartitioner, CGPartitioner
from ReachMM.utils import run_times
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import shapely.geometry as sg
import shapely.ops as so
import polytope

x1, x2, u, w = sp.symbols('x1 x2 u w')

f_eqn = [
    x1 + x2 + 0.5*u,
    x2 + u
]
# f_eqn = [
#     x2 + 0.5*u,
#     u
# ]

t_spec = DiscreteTimeSpec()
# t_spec = ContinuousTimeSpec(1,1)
t_end = 5; tt = t_spec.tt(0,t_end)
sys = System([x1, x2], [u], [w], f_eqn, t_spec)
net = NeuralNetwork('models/10r5r1')
clsys = NNCSystem(sys, net, NNCSystem.InclOpts('jacobian'))
print(clsys)

# partitioner = UniformPartitioner(clsys)
# popts = UniformPartitioner.Opts(6,2)
# unipartitioner = UniformPartitioner(clsys)
# cgpartitioner  = CGPartitioner(clsys)
popts = [CGPartitioner.Opts(0.1,3,1,1), CGPartitioner.Opts(0.05,6,2,1)]
# popts = [UniformPartitioner.Opts(0,0), UniformPartitioner.Opts(6,2)]

x0 = np.array([
    np.interval(2.5,3.0),
    np.interval(-0.25,0.25)
])

fig, axs = plt.subplots(1,2,dpi=100,figsize=[11,5],squeeze=True)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.125, hspace=0.2)

mc_trajs = clsys.compute_mc_trajectories(0,t_end,x0,200)

for axi, ax in enumerate(axs) :
    popt = popts[axi]
    if type(popt) is CGPartitioner.Opts :
        partitioner = CGPartitioner(clsys)
    elif type(popt) is UniformPartitioner.Opts :
        partitioner = UniformPartitioner(clsys)

    def run () :
        return partitioner.compute_reachable_set(0, t_end, x0, popt)
    rs, runtimes = run_times(args.runtime_N, run)
    print(f'Runtime: {np.mean(runtimes)} \\pm {np.std(runtimes)}')
    print(f'Area: {rs.area(t_end)}')

    rs.draw_rs(ax, tt, lw=1)

    ax.set_xlim([-1,3.5])
    ax.set_ylim([-1.5,1])

    for traj in mc_trajs :
        traj.scatter2d(axs[0], tt, s=0.25, c='r')
        traj.scatter2d(axs[1], tt, s=0.25, c='r')
    
    LP_parts = ['4.npy', '16.npy']
    # LP_parts = ['55.npy', '205.npy']
    LipBnB_eps = ['0.1.npy', '0.001.npy']
    # ReachLP-Uniform
    # LP = np.load('comparisons/ReachLP-Results/GreedySimGuided-' + LP_parts[axi], allow_pickle=True)
    LP = np.load('comparisons/ReachLP-Results/Uniform-' + LP_parts[axi], allow_pickle=True)
    LP_rs = np.array([a[1] for a in LP])
    print('ReachLP Areas for Uniform-' + LP_parts[axi])
    for t in tt[1:6] :
        boxes = [sg.box(box[0,0],box[1,0],box[0,1],box[1,1]) for box in LP_rs[:,t-1,:,:]]
        shape = so.unary_union(boxes)
        xs, ys = shape.exterior.xy    
        ax.fill(xs, ys, alpha=1, fc='none', ec='tab:orange', lw=1)
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

fig.savefig('figures/ojcsys2023/DI.pdf')

plt.show()
