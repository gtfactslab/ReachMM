import numpy as np
import interval
from interval import from_cent_pert, get_lu, get_cent_pert
import sympy as sp
from ReachMM.time import *
from ReachMM.system import *
from ReachMM.reach import UniformPartitioner, CGPartitioner
from ReachMM.control import ConstantDisturbance
from ReachMM.utils import run_times, draw_iarray
import matplotlib.pyplot as plt

x1, x2, x3, x4, u, w = sp.symbols('x1 x2 x3 x4 u w')
x_vars = [x1, x2, x3, x4]

f_eqn = [
    x2,
    -x1 + 0.1*sp.sin(x3),
    x4,
    11*u
]
# spec = (Drel - (Dsafe := (Ddefault:=10) + Tgap*vego))
# print(spec)
# spec_lam = sp.lambdify((x_vars,), spec, 'numpy')

t_spec = ContinuousTimeSpec(0.05,0.5)
# t_spec = DiscretizedTimeSpec(0.05)
sys = System(x_vars, [u], [w], f_eqn, t_spec)
net = NeuralNetwork('models/nn_tora_relu_tanh')
# del(net.seq[-1])
# del(net.seq[-1])
print(net.seq)
clsys = NNCSystem(sys, net, 'jacobian')
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
    np.interval(-0.3,-0.28)
])
xcent, xpert = get_cent_pert(x0)
# print(net.seq[0].weight.detach().numpy() @ xcent + net.seq[0].bias.detach().numpy())

# partitioner = UniformPartitioner(clsys)
# popts = UniformPartitioner.Opts(1, 1)
partitioner = CGPartitioner(clsys)
popts = CGPartitioner.Opts(0.25, 0.1, 1, 1)

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

# clsys.sys.t_spec = ContinuousTimeSpec(0.01,0.5)
trajs = clsys.compute_mc_trajectories(0,t_end,x0,100)
# print(traj(t_end))
tt = clsys.sys.t_spec.tt(0,t_end)
for traj in trajs :
    plt.plot(traj(tt)[:,0], traj(tt)[:,1], color='tab:red')

print(rs(tt)[:,2])
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

