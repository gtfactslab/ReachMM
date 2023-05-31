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

sx, sy, sxd, syd, u1, u2, w = sp.symbols('sx sy sxd syd u1 u2 w')
x_vars = [sx, sy, sxd, syd]
u_vars = [u1, u2]

m = 12
n = 0.001027
f_eqn = [
    sxd,
    syd,
    3*n**2*sx + 2*n*syd + u1/m,
    -2*n*syd + u2/m
]
spec = ((0.2 + 2*n*sp.sqrt(sx**2 + sy**2)) - sp.sqrt(sxd**2 + syd**2))
print(spec)
spec_lam = sp.lambdify((x_vars,), spec, 'numpy')

t_spec = ContinuousTimeSpec(0.1,1)
# t_spec = DiscretizedTimeSpec(0.05)
sys = System(x_vars, u_vars, [w], f_eqn, t_spec)
net = NeuralNetwork('models/model')
# del(net.seq[-1])
# del(net.seq[-1])
clsys = NNCSystem(sys, net, 'interconnect')
print(clsys)
t_end = 40

# x0 = np.array([
#     np.interval(0.6,0.7),
#     np.interval(-0.7,-0.6),
#     np.interval(-0.4,-0.3),
#     np.interval(0.5,0.6)
# ])
x0 = np.array([
    np.interval(70,74),
    np.interval(70,74),
    np.interval(0.24,0.28),
    np.interval(0.24,0.28)
    # np.interval(87,89),
    # np.interval(87,89),
    # np.interval(-0.01,0.01),
    # np.interval(-0.01,0.01)
    # np.interval(70,106),
    # np.interval(70,106),
    # np.interval(-0.28,0.28),
    # np.interval(-0.28,0.28)
])
xcent, xpert = get_cent_pert(x0)
# print(net.seq[0].weight.detach().numpy() @ xcent + net.seq[0].bias.detach().numpy())

partitioner = UniformPartitioner(clsys)
popts = UniformPartitioner.Opts(0,0)
# partitioner = CGPartitioner(clsys)
# popts = CGPartitioner.Opts(0.25, 0.1, 1, 1)

tt = t_spec.tt(0,t_end)

def run () :
    rs = partitioner.compute_reachable_set(0,t_end,x0,popts)
    safe = rs.check_safety(spec_lam, tt)
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

