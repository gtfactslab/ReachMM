import numpy as np
import interval
from interval import from_cent_pert, get_lu, get_cent_pert
from inclusion import Ordering
import sympy as sp
from ReachMM.time import *
from ReachMM.system import *
from ReachMM.reach import UniformPartitioner, CGPartitioner
from ReachMM.control import ConstantDisturbance
from ReachMM.utils import run_times, draw_iarray
import matplotlib.pyplot as plt
import torch

x1, x2, x3, x4, y1, y2, u1, u2, w = sp.symbols('x1 x2 x3 x4 y1 y2 u1 u2 w')
x_vars = [x1, x2, x3, x4, y1, y2]
u_vars = [u1, u2]

f_eqn = [
    x4*sp.cos(x3),
    x4*sp.sin(x3),
    u2,
    u1 + w,
    x4*(sp.cos(x3) + sp.sin(x3)),
    x4*(sp.cos(x3) - sp.sin(x3)),
]
# spec = (Drel - (Dsafe := (Ddefault:=10) + Tgap*vego))
# print(spec)
# spec_lam = sp.lambdify((x_vars,), spec, 'numpy')

# t_spec = ContinuousTimeSpec(0.05,0.2)
t_spec = ContinuousTimeSpec(0.2,0.2)
# t_spec = DiscretizedTimeSpec(0.2)

ref = AffineRefine(
    M = np.array([
        [-1,-1, 0, 0, 1, 0],
        [-1, 1, 0, 0, 0, 1]
    ]), 
    b = np.array([0, 0])
)
sys = System(x_vars, u_vars, [w], f_eqn, t_spec)
net = NeuralNetwork('models/controllerB_nnv')

g_lin = torch.nn.Linear(6,4,False)
g_lin.weight = torch.nn.Parameter(
    torch.tensor(np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
    ]).astype(np.float32))
)
net.seq.insert(0, g_lin)
clsys = NNCSystem(sys, net, 
                  NNCSystem.InclOpts('interconnect'),
                #   NNCSystem.InclOpts('jacobian+interconnect',
                #                      orderings=[Ordering((0,1,2,3,4,5))]),
                  dist=ConstantDisturbance([0], [np.interval(-0.001,0.001)]))
clsys.set_four_corners()
t_end = 3

x0 = np.array([
    np.interval(9.5,9.55),
    np.interval(-4.5,-4.45),
    np.interval(2.1,2.11),
    np.interval(1.5,1.51),
    np.interval(9.5,9.55) + np.interval(-4.5,-4.45),
    np.interval(9.5,9.55) - np.interval(-4.5,-4.45),
])
xcent, xpert = get_cent_pert(x0)
# print(net.seq[0].weight.detach().numpy() @ xcent + net.seq[0].bias.detach().numpy())

partitioner = UniformPartitioner(clsys)
popts = UniformPartitioner.Opts(0,0)
# partitioner = CGPartitioner(clsys)
# popts = CGPartitioner.Opts(0.06, 0.5, 3, 1)

tt = t_spec.tt(0,t_end)

def run () :
    rs = partitioner.compute_reachable_set(0,t_end,x0,popts)
    safe = 'T' # safe = rs.check_safety(spec_lam, tt)
    return rs, safe
(rs, safe), times = run_times(1, run)

print(f'Safe: {safe} in {np.mean(times)} \\pm {np.std(times)} (s)')

xx = rs(tt)
print(rs(t_end))
rs.draw_rs(plt, tt)

trajs = clsys.compute_mc_trajectories(0,t_end,x0,100)
# print(traj(t_end))
tt = clsys.sys.t_spec.tt(0,t_end)
for traj in trajs :
    plt.plot(traj(tt)[:,0], traj(tt)[:,1], color='tab:red')

# fig, ax = plt.subplots(1, 1, squeeze=True)

# Drel_xx  = sp.lambdify((x_vars,), Drel , 'numpy')(xx)
# Drel_l, Drel_u = get_lu(Drel_xx)
# Dsafe_xx = sp.lambdify((x_vars,), Dsafe, 'numpy')(xx)
# Dsafe_l, Dsafe_u = get_lu(Dsafe_xx)

# pltl = ax.plot(tt, Drel_l, color='tab:blue')
# pltu = ax.plot(tt, Drel_u, color='tab:blue')
# ax.fill_between(tt, Drel_l, Drel_u, color='tab:blue', alpha=0.25)
# pltl = ax.plot(tt, Dsafe_l, color='tab:red')
# pltu = ax.plot(tt, Dsafe_u, color='tab:red')
# ax.fill_between(tt, Dsafe_l, Dsafe_u, color='tab:red', alpha=0.25)
plt.show()

