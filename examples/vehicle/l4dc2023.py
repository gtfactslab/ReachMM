import argparse
parser = argparse.ArgumentParser(description="Vehicle (bicycle model) Experiments for L4DC 2023 Paper")
parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",\
                    type=int, default=1)
args = parser.parse_args()

import numpy as np
import interval
from interval import from_cent_pert
import sympy as sp

from ReachMM import ContinuousTimeSpec
from ReachMM import System, NeuralNetwork, NNCSystem
from ReachMM import UniformPartitioner, CGPartitioner
from ReachMM.utils import run_times
import matplotlib.pyplot as plt

px, py, psi, v, u1, u2, w = sp.symbols('p_x p_y psi v u1 u2 w')

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

t_spec = ContinuousTimeSpec(0.01,0.25)
sys = System([px, py, psi, v], [u1, u2], [w], f_eqn, t_spec)
net = NeuralNetwork('models/100r100r2')
clsys = NNCSystem(sys, net, 'interconnect')
# clsys = NNCSystem(sys, net, 'jacobian')

t_span = [0,1.25]

cent = np.array([8,8,-2*np.pi/3,2])
pert = np.array([0.1,0.1,0.01,0.01])
x0 = from_cent_pert(cent, pert)

# partitioner = UniformPartitioner(clsys)
partitioner = CGPartitioner(clsys)

# Experiment 1
opts = CGPartitioner.Opts(0.25, 0.1, 2, 0)
# opts = UniformPartitioner.Opts(1,0)
rs, times = run_times(args.runtime_N, partitioner.compute_reachable_set, t_span[0], t_span[1], x0, opts)

print(np.mean(times), '\pm', np.std(times))

rs.draw_rs(plt, t_spec.tt(t_span[0], t_span[1]), color='r')
plt.show()