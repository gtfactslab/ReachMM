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
from ReachMM import System, NeuralNetwork, NNCSystem, NeuralNetworkControl
from ReachMM import UniformPartitioner, CGPartitioner
from ReachMM.utils import run_times
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import shapely.geometry as sg
import shapely.ops as so
import polytope
import torch

x1, x2, u, w = sp.symbols('x1 x2 u w')

# A = np.array([[0,1],[0,0]])
# B = np.array([[0],[1]])
# A = np.array([[1,1],[0,1]])
# B = np.array([[0.5],[1]])
A = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
B = np.array([[0,0],[1,0],[0,0],[0,1]])

# f_eqn = [
#     x1 + x2 + 0.5*u,
#     x2 + u
# ]
# net = NeuralNetwork('models/10r5r1')
net = NeuralNetwork('../platooning/models/100r100r2')
# net = NeuralNetwork('models/close_linear')

# K = np.array([[-0.24977, -0.69873]])
# K = np.array([[-0.48963, -0.96968]])
# K = np.array([[-0.638  0.43499]])
# K = np.array([[-0.40741, -0.90137]])
# K = np.array([[-0.47133, -0.17458]])
# K = np.array([[-1.6, -5.1]])
# K = [[-0.2081, -0.008447]]

# x0 = np.array([
#     np.interval(-0.01,0.01),
#     np.interval(-0.01,0.01),
#     np.interval(-0.01,0.01),
#     np.interval(-0.01,0.01)
# ])
# nnc = NeuralNetworkControl(net)
# nnc.prime(x0)
# nnc.step(0,x0)
# print(nnc._C)
# K = nnc._C

# input()

K = (torch.autograd.functional.jacobian(net,torch.zeros(4))).cpu().detach().numpy()
print('K: ', K)
Acl = A + B@K
# print(Acl)
# print(K)
L, U = np.linalg.eig(Acl)
print(L)
print(U)

input()

# T = U
T = np.array([-np.real(U[:,0]),np.imag(U[:,0])]).T
# T = np.array([[1,2],[1,0]])
Tinv = np.linalg.inv(T)
temp = T
T = Tinv
Tinv = temp
print('T: ')
print(T)
print('Tinv: ')
print(Tinv)

print('K*Tinv')
print(K@Tinv)

# net.seq.insert(0, torch.nn.Linear(2,2,False))
# net.seq.append(torch.nn.Linear(2,2,False))
# net[0].weight = torch.nn.Parameter(torch.tensor(Tinv.astype(np.float32)))
# net[0].bias = torch.nn.Parameter(torch.tensor([0,0],dtype=torch.float32))
# net[-1].weight = torch.nn.Parameter(torch.tensor(T))

A = T@(A)@Tinv
print('A transformed: ')
print(A)
B = T@B
# f_eqn = sp.Matrix(A)@sp.Matrix([x1,x2]) + sp.Matrix(B)@K@Tinv@sp.Matrix([x1,x2]) #+ sp.Matrix(B)@sp.Matrix([u])
f_eqn = sp.Matrix(A)@sp.Matrix([x1,x2]) + sp.Matrix(B)@sp.Matrix([u])
# f_eqn = [f_mult[0], f_mult[1]]
# f_eqn = sp.Matrix(A)@sp.Matrix([x1,x2]) + sp.Matrix(B)@sp.Matrix(K)@sp.Matrix([x1,x2])
# f_eqn = sp.Matrix(A)@sp.Matrix([x1,x2])
# print(f_eqn)

# t_spec = DiscreteTimeSpec()
t_spec = ContinuousTimeSpec(0.1,0.1)
t_span = [0,5]
tt = t_spec.tt(*t_span)
sys = System([x1, x2], [u], [w], f_eqn, t_spec)
clsys = NNCSystem(sys, net, NNCSystem.InclOpts('jacobian'))
partitioner = UniformPartitioner(clsys)
part_opts = UniformPartitioner.Opts(0,0)

print(clsys)

# x0 = T @ np.array([
#     np.interval(-0.1,0.1),
#     np.interval(-0.1,0.1)
# ])
# x0 = T @ np.array([
#     np.interval(2.5,3.0),
#     np.interval(-0.25,0.25)
# ])
# x0 = np.array([
#     np.interval(-0.1,0.1),
#     np.interval(-0.1,0.1)
# ])
x0 = np.array([
    np.interval(-1,1),
    np.interval(-1,1)
])
print(net.seq[0].bias)
print(x0)

rs = partitioner.compute_reachable_set(*t_span, x0, part_opts)

rs.draw_rs(plt, tt)
# polytope.Polytope()

mc_trajs = clsys.compute_mc_trajectories(*t_span, x0, 1000)
for mc_traj in mc_trajs :
    mc_traj.plot2d(plt, tt)
# traj = clsys.compute_trajectory(*t_span, np.array([0.0,0.005]))
# traj.plot2d(plt, tt)

plt.show()

