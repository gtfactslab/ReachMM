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
from ReachMM import System, NeuralNetwork, NNCSystem, NeuralNetworkControl, NoDisturbance, ConstantDisturbance
from ReachMM import UniformPartitioner, CGPartitioner
from ReachMM.utils import run_times
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import shapely.geometry as sg
import shapely.ops as so
import polytope
import torch

A = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
B = np.array([[0,0],[1,0],[0,0],[0,1]])
Q = np.array([ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ])
R = np.array([ [0.5,0], [0,0.5] ])

px, vx, py, vy = (x_vars := sp.symbols('px vx py vy'))
ux, uy = (u_vars := sp.symbols('ux uy'))
wx, wy = (w_vars := sp.symbols('wx wy'))
f_eqn = A@x_vars + B@u_vars
t_spec = ContinuousTimeSpec(0.1, 0.1)
sys = System(x_vars, u_vars, w_vars, f_eqn, t_spec)
print(sys)
net = NeuralNetwork('models/100r100r2_MPC')
clsys = NNCSystem(sys, net, NNCSystem.InclOpts('jacobian'), dist=NoDisturbance(2))

onetraj = clsys.compute_trajectory(0,100,np.zeros(4))
xeq = onetraj(100)
print(xeq)

x0 = xeq + np.array([
    np.interval(-0.05,0.05),
    np.interval(-0.05,0.05),
    np.interval(-0.05,0.05),
    np.interval(-0.05,0.05)
])
nnc = NeuralNetworkControl(net)
nnc.prime(x0)
nnc.step(0,x0)
K = nnc._C
# K = (torch.autograd.functional.jacobian(net,torch.zeros(4))).cpu().detach().numpy()
print('K: ', K)
Acl = A + B@K
# print(Acl)
# print(K)
L, U = np.linalg.eig(Acl)
print(L)
print(U)

# T = U
T = np.array([-np.real(U[:,0]),np.imag(U[:,0]),-np.real(U[:,2]),np.imag(U[:,2])]).T
# T = np.array([[1,2],[1,0]])
# T = np.eye(4)
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

net.seq.insert(0, torch.nn.Linear(4,4))
# net.seq.append(torch.nn.Linear(2,2,False))
net[0].weight = torch.nn.Parameter(torch.tensor(Tinv.astype(np.float32)))
net[0].bias = torch.nn.Parameter(torch.tensor([0,0,0,0],dtype=torch.float32))
# net[-1].weight = torch.nn.Parameter(torch.tensor(T))

A = T@(A)@Tinv
print('A transformed: ')
print(A)
B = T@B
# f_eqn = sp.Matrix(A)@sp.Matrix([x1,x2]) + sp.Matrix(B)@K@Tinv@sp.Matrix([x1,x2]) #+ sp.Matrix(B)@sp.Matrix([u])
f_eqn = A@x_vars + B@u_vars

t_spec = ContinuousTimeSpec(0.1,0.1)
t_span = [0,0.1]
# t_span = [0,10]
tt = t_spec.tt(*t_span)
sys = System(x_vars, u_vars, w_vars, f_eqn, t_spec)
dist_int = np.array( [np.interval(-0.01,0.01), np.interval(-0.01,0.01)] )
dist_cent, dist_pert = interval.get_cent_pert(dist_int)
clsys = NNCSystem(sys, net, NNCSystem.InclOpts('jacobian'), dist=ConstantDisturbance(dist_cent, dist_int))
partitioner = UniformPartitioner(clsys)
part_opts = UniformPartitioner.Opts(0,0)

print(clsys)

# x0 = np.array([
#     np.interval(-0.1,0.1),
#     np.interval(-0.1,0.1),
#     np.interval(-0.3,0.7),
#     np.interval(-0.3,0.7)
# ])
x0 = T@xeq + np.array([
    np.interval(-0.05,0.05),
    np.interval(-0.05,0.05),
    np.interval(-0.07,0.07),
    np.interval(-0.1,0.1)
])
# x0 = T@xeq + np.array([
#     np.interval(-10,10),
#     np.interval(-1,1),
#     np.interval(-10,10),
#     np.interval(-1,1)
# ])
print(net.seq[0].bias)
print(x0)

fig, axs = plt.subplots(1,2,figsize=[8,4])

rs = partitioner.compute_reachable_set(*t_span, x0, part_opts)

rs.draw_rs(axs[0], tt, 0, 1)
rs.draw_rs(axs[1], tt, 2, 3)
print(rs(t_span[0]))
print(rs(t_span[1]))
print(np.subseteq(rs(t_span[1]), rs(t_span[0])))

# polytope.Polytope()

mc_trajs = clsys.compute_mc_trajectories(*t_span, x0, 200)
for mc_traj in mc_trajs :
    mc_traj.plot2d(axs[0], tt, 0, 1, zorder=0)
    mc_traj.plot2d(axs[1], tt, 2, 3, zorder=0)
# traj = clsys.compute_trajectory(*t_span, np.array([0.0,0.005]))
# traj.plot2d(plt, tt)

# axs[0].add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon',zorder=-1))
# axs[0].add_patch(Circle((-4,4),3/1.25,lw=0,fc='salmon',zorder=-1))
# axs[0].add_patch(Circle((-4,-4),3/1.25,lw=0,fc='salmon',zorder=-1))
# axs[0].add_patch(Circle((4,-4),3/1.25,lw=0,fc='salmon',zorder=-1))
        
plt.show()

