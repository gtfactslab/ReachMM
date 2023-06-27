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
import torch

x1, x2, x3, x4, y1, y2, u, w = sp.symbols('x1 x2 x3 x4 y1 y2 u w')
x_vars = [x1, x2, x3, x4, y1, y2]

f_eqn = [
    x2,
    -x1 + 0.1*sp.sin(x3),
    x4,
    11*sp.tanh(u),
    x2 - x1 + 0.1*sp.sin(x3),
    x2 + x1 - 0.1*sp.sin(x3),
]
# spec = (Drel - (Dsafe := (Ddefault:=10) + Tgap*vego))
specs = [ (x1 + 0.1), (0.2 - x1), (x2 + 0.9), (-0.6 - x2) ]
# print(spec)
spec_lam = [sp.lambdify((x_vars,), spec, 'numpy') for spec in specs]

# t_spec = ContinuousTimeSpec(0.05,0.5)
t_spec = ContinuousTimeSpec(0.005,0.5)
# t_spec = DiscretizedTimeSpec(0.05)
ref = AffineRefine(
    M = np.array([
        [-1, -1, 0, 0, 1, 0],
        [-1,  1, 0, 0, 0, 1],
    ]),
    b = np.array([ 0,0 ])
)
sys = System(x_vars, [u], [w], f_eqn, t_spec, ref)
net = NeuralNetwork('models/nn_tora_relu_tanh')
del(net.seq[-1])
del(net.seq[-1])
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
print(net.seq)
clsys = NNCSystem(sys, net, incl_opts=
                  NNCSystem.InclOpts('jacobian+interconnect', 
                #   NNCSystem.InclOpts('interconnect', 
                                     orderings=[Ordering((0,1,2,3,4,5))]))
                #   g_tuple=(x_vars,), g_eqn=[x1, x2, x3, x4])
clsys.set_four_corners()
t_end = 5

# x0 = np.array([
#     np.interval(0.6,0.7),
#     np.interval(-0.7,-0.6),
#     np.interval(-0.4,-0.3),
#     np.interval(0.5,0.6)
# ])
x0 = np.array([
    np.interval(-0.77,-0.75),
    np.interval(-0.45,-0.43),
    np.interval(0.51,0.54),
    np.interval(-0.3,-0.28),
    np.interval(-0.77,-0.75) + np.interval(-0.45,-0.43),
    np.interval(-0.77,-0.75) - np.interval(-0.45,-0.43),
])
xcent, xpert = get_cent_pert(x0)
# print(net.seq[0].weight.detach().numpy() @ xcent + net.seq[0].bias.detach().numpy())

partitioner = UniformPartitioner(clsys)
popts = UniformPartitioner.Opts(0, 0)
# partitioner = CGPartitioner(clsys)
# popts = CGPartitioner.Opts(0.25, 0.1, 1, 1)

tt = t_spec.tt(0,t_end)

def run () :
    rs = partitioner.compute_reachable_set(0,t_end,x0,popts)
    safe = rs.check_safety_tt(spec_lam, tt)
    print(safe)
    return rs, safe
(rs, safe), times = run_times(1, run)

print(f'Safe: {safe} in {np.mean(times)} \\pm {np.std(times)} (s)')

xx = rs(tt)
print(rs(t_end))
rs.draw_rs(plt, tt)

# clsys.sys.t_spec = ContinuousTimeSpec(0.01,0.5)
trajs = clsys.compute_mc_trajectories(0,t_end,x0,100)
# print(traj(t_end))
tt = clsys.sys.t_spec.tt(0,t_end)
for traj in trajs :
    plt.plot(traj(tt)[:,0], traj(tt)[:,1], color='tab:red')

plt.xlim([-0.1,0.2])
plt.ylim([-0.9,-0.6])
# print(rs(tt)[:,2])
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

