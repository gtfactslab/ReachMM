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
import torch

eps = 0.1
kp = -1.6
kd = -5.1

x1, x2, u, w = sp.symbols('x1 x2 u w')

# f_eqn = [
#     x1 + x2 + 0.5*u,
#     x2 + u
# ]
f_eqn = [
    x2,
    u
]

# t_spec = DiscreteTimeSpec()
t_spec = ContinuousTimeSpec(0.001,0.001)
t_end = 100; tt = t_spec.tt(0,t_end)
sys = System([x1, x2], [u], [w], f_eqn, t_spec)

net = NeuralNetwork('models/close_linear', False)
print(net)

net.seq[0].weight = torch.nn.Parameter(torch.tensor([[1,0],[-1,0],[0,1],[0,-1]],dtype=torch.float32))
net.seq[0].bias = torch.nn.Parameter(torch.tensor([-eps,-eps,-eps,-eps],dtype=torch.float32))
net.seq[2].weight = torch.nn.Parameter(torch.tensor([ [kp, -kp, kd, -kd] ],dtype=torch.float32))
net.seq[2].bias = torch.nn.Parameter(torch.tensor([0],dtype=torch.float32))

print(net(torch.tensor([0,-1],dtype=torch.float32)))

net.save()

# net = NeuralNetwork('models/10r5r1')
clsys = NNCSystem(sys, net, NNCSystem.InclOpts('jacobian'))
# print(clsys)

# cent = np.array([0.00012,0.0])
# # cent = np.array([0.0,0.0])
# pert = np.array([-100,100])
# x0 = from_cent_pert(cent, pert)

# print(clsys.control.u(0,cent))

# clsys.control.prime(x0)
# clsys.control.step(0, x0)
# clsys.prime(x0)
# print(clsys._l_jac_cont(0, x0))

# # traj = clsys.compute_trajectory(0,t_end,np.array([0.0,0.0]))
# # print(traj(tt))

# # x0 = np.array([
# #     np.interval(0,0),
# #     np.interval(0,0)
# # ])


