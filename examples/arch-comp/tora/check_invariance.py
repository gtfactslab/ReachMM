import argparse
parser = argparse.ArgumentParser(description="Vehicle (bicycle model) Experiments for L4DC 2023 Paper")
parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",\
                    type=int, default=1)
args = parser.parse_args()

import numpy as np
import interval
from interval import from_cent_pert, get_lu, get_cent_pert
from inclusion import Corner, Ordering
import sympy as sp
from ReachMM.time import *
from ReachMM.system import *
from ReachMM.reach import UniformPartitioner, CGPartitioner
from ReachMM.control import ConstantDisturbance
from ReachMM.utils import run_times, draw_iarray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

x1, x2, x3, x4 = (x_vars := sp.symbols('x1 x2 x3 x4'))
u, w = sp.symbols('u w')

f_eqn = [
    x2,
    -x1 + 0.1*sp.sin(x3),
    x4,
    11*sp.tanh(u),
]

t_spec = ContinuousTimeSpec(0.005,0.5)
sys = System(x_vars, [u], [w], f_eqn, t_spec)
net = NeuralNetwork('models/nn_tora_relu_tanh')
# Remove final tanh and offset layer.
del(net.seq[-1])
del(net.seq[-1])
print(net.seq)
clsys = NNCSystem(sys, net, incl_opts=NNCSystem.InclOpts('jacobian+interconnect'))
clsys.set_standard_ordering()
clsys.set_four_corners()
t_end = 5

print(clsys)

onetraj = clsys.compute_trajectory(0,100,np.zeros(4))
xeq = onetraj(100)
# xeq = np.zeros(4)

Aeq, Beq, Deq = sys.get_ABD(xeq, clsys.control.u(0,xeq), [0])

x0 = xeq + np.array([
    np.interval(-0.5,0.5),
    np.interval(-0.5,0.5),
    np.interval(-0.5,0.5),
    np.interval(-0.5,0.5)
])
nnc = NeuralNetworkControl(net)
nnc.prime(x0)
nnc.step(0,x0)
K = nnc._C
print('K: ', K)
Acl = Aeq + Beq@K
L, U = np.linalg.eig(Acl)
print(L)
print(U)

Tinv = np.array([-np.real(U[:,0]),np.imag(U[:,0]),-np.real(U[:,2]),np.imag(U[:,2])]).T
# Tinv = np.eye(4)
T = np.linalg.inv(Tinv)
print('T: '); print(T)
print('Tinv: '); print(Tinv)

net.seq.insert(0, torch.nn.Linear(4,4))
net[0].weight = torch.nn.Parameter(torch.tensor(Tinv.astype(np.float32)))
net[0].bias = torch.nn.Parameter(torch.tensor([0,0,0,0],dtype=torch.float32))

y1, y2, y3, y4 = (y_vars := sp.symbols('y1 y2 y3 y4'))
print(y_vars)

xr1, xr2, xr3, xr4 = tuple(Tinv@sp.Matrix(y_vars))

g_eqn = sp.Matrix(f_eqn)
g_eqn = g_eqn.subs(x1, xr1).subs(x2, xr2).subs(x3, xr3).subs(x4, xr4)
g_eqn = T@g_eqn
print(g_eqn)

t_spec = ContinuousTimeSpec(0.1,0.1)
t_span = [0,1]
tt = t_spec.tt(*t_span)
sys = System(y_vars, [u], [w], g_eqn, t_spec)
clsys = NNCSystem(sys, net, incl_opts=NNCSystem.InclOpts('jacobian+interconnect'))
clsys.set_standard_ordering()
clsys.set_four_corners()

partitioner = UniformPartitioner(clsys)
part_opts = UniformPartitioner.Opts(0,0)

x0 = T@xeq + np.array([
    np.interval(-1,1),
    np.interval(-1,1),
    np.interval(-2,2),
    np.interval(-2,2)
])

fig, axs = plt.subplots(1,2,figsize=[8,4])

# rs = partitioner.compute_reachable_set(*t_span, x0, part_opts)
# rs.draw_rs(axs[0], tt, 0, 1)
# rs.draw_rs(axs[1], tt, 2, 3)
# print(rs(t_span[0]))
# print(rs(t_span[1]))
# print(np.subseteq(rs(t_span[1]), rs(t_span[0])))

mc_trajs = clsys.compute_mc_trajectories(*t_span, x0, 1000)
for mc_traj in mc_trajs :
    xx = Tinv@mc_traj(tt).T
    # mc_traj.plot2d(axs[0], tt, 0, 1, zorder=0)
    # mc_traj.plot2d(axs[1], tt, 2, 3, zorder=0)
    axs[0].plot(xx[0,:], xx[1,:], zorder=0)
    axs[1].plot(xx[2,:], xx[3,:], zorder=0)

plt.show()

