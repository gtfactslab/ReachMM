import numpy as np
import interval
from interval import from_cent_pert
import sympy as sp
from ReachMM.time import *
from ReachMM.system import *
from ReachMM.reach import UniformPartitioner, CGPartitioner
from ReachMM.utils import run_times, gen_ics_iarray
import matplotlib.pyplot as plt

def _benchmark1 (N) :
    x1, x2, u, w = sp.symbols('x1 x2 u w')
    f_eqn = [
        x2,
        u*x2**2 - x1
    ]
    t_spec = ContinuousTimeSpec(0.05,0.2)
    sys = System([x1, x2], [u], [w], f_eqn, t_spec)
    net = NeuralNetwork('models/nn_1_relu')
    clsys = NNCSystem(sys, net, 'interconnect')
    # t_end = 7
    t_end = 5
    x0 = np.array([ np.interval(0.8,0.9), np.interval(0.5,0.6) ])
    # opts = CGPartitioner.Opts(0.5, 0.1, 2, 0, -1,-1,-1, True)
    # partitioner = CGPartitioner(clsys)
    opts = UniformPartitioner.Opts(3, 0)
    partitioner = UniformPartitioner(clsys)
    rs, times = run_times(N, partitioner.compute_reachable_set, 0,t_end,x0,opts)

    tt = t_spec.tt(0, t_end)
    print(rs(tt))
    print(np.mean(times), '\pm', np.std(times))
    avg_runtime, std_runtime = np.mean(times), np.std(times)

    plt.text(-0.25,0.55,f'runtime: ${avg_runtime:.3f}\pm{std_runtime:.3f}$',fontsize=15,verticalalignment='top')
    # xx, times = run_times(10, clsys.compute_trajectory, 0, t_end, x0)

    # rs = xx(tt)
    MC_N = 100
    trajs = clsys.compute_mc_trajectories(0, t_end, x0, MC_N)
    for traj in trajs :
        traj.plot2d(plt, tt, c='tab:red', alpha=1, zorder=0)
    rs.draw_rs(plt, t_spec.tt(0,t_end)[::2], color='tab:blue')
    plt.savefig('benchmark1.pdf')
    plt.show()

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark 1")
    parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",\
                        type=int, default=1)
    args = parser.parse_args()

    _benchmark1(args.runtime_N)
