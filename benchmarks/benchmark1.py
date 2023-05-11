import numpy as np
import interval
from interval import from_cent_pert
import sympy as sp
from ReachMM.time import *
from ReachMM.system import *
from ReachMM.reach import UniformPartitioner
from ReachMM.utils import run_times, draw_iarrays
import matplotlib.pyplot as plt

def _benchmark1 (N) :
    x1, x2, u, w = sp.symbols('x1 x2 u w')
    f_eqn = [
        x2,
        u*x2**2 - x1
    ]

    t_spec = ContinuousTimeSpec(0.025,0.2)
    # t_spec = DiscretizedTimeSpec(0.05)
    # t_spec = DiscreteTimeSpec()
    sys = NLSystem([x1, x2], [u], [w], f_eqn, t_spec)
    net = NeuralNetwork('models/nn_1_relu')
    clsys = NNCSystem(sys, net, 'jacobian')
    # clsys = NNCSystem(sys, net, 'interconnect')

    t_end = 7

    traj = clsys.compute_trajectory(0, t_end, np.array([ 0.85, 0.55 ]))
    # x0 = np.array([ np.interval(0.825,0.85), np.interval(0.525,0.55) ])
    x0 = np.array([ np.interval(0.8,0.9), np.interval(0.5,0.6) ])
    # x0 = np.array([ np.interval(0.849,0.851), np.interval(0.549,0.551) ])

    partitioner = UniformPartitioner(clsys)
    rs, times = run_times(1, partitioner.compute_reachable_set,0,t_end,x0,0,0)
    tt = t_spec.tt(0, t_end)
    print(rs(tt))
    print(np.mean(times), '\pm', np.std(times))

    # xx, times = run_times(10, clsys.compute_trajectory, 0, t_end, x0)

    # rs = xx(tt)
    plt.plot(traj(tt)[:,0], traj(tt)[:,1])
    # plt.scatter(traj(tt)[:,0], traj(tt)[:,1],s=5)
    # draw_iarrays(plt, rs(tt))
    rs.draw_rs(plt, t_spec.uu(0,t_end))
    plt.show()

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark 1")
    parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",\
                        type=int, default=1)
    args = parser.parse_args()

    _benchmark1(args.runtime_N)
