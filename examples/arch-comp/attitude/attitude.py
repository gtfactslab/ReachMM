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

# States
w1, w2, w3, p1, p2, p3 = sp.symbols('w1 w2 w3 p1 p2 p3')
x_vars = [w1, w2, w3, p1, p2, p3]
# Control
u1, u2, u3 = sp.symbols('u1 u2 u3')
u_vars = [u1, u2, u3]
# Disturbance
w_dist = sp.symbols('w_dist')

f_eqn = [
    0.25*(u1+w2*w3),
    0.5*(u2-3*w1*w3),
    u3 + 2*w1*w2,
    0.5*(w2*(p1**2 + p2**2 + p3**2 - p3) + w3*(p1**2 + p2**2 + p2 + p3**2) + w1*(p1**2 + p2**2 + p3**2 + 1)),
    0.5*(w1*(p1**2 + p2**2 + p3**2 + p3) + w3*(p1**2 - p1 + p2**2 + p3**2) + w2*(p1**2 + p2**2 + p3**2 + 1)),
    0.5*(w1*(p1**2 + p2**2 - p2 + p3**2) + w2*(p1**2 + p1 + p2**2 + p3**2) + w3*(p1**2 + p2**2 + p3**2 + 1))
]
# spec = (Drel - (Dsafe := (Ddefault:=10) + Tgap*vego))
# print(spec)
# spec_lam = sp.lambdify((x_vars,), spec, 'numpy')

t_spec = ContinuousTimeSpec(0.1,0.1)
# t_spec = DiscretizedTimeSpec(0.1)
sys = System(x_vars, u_vars, [w_dist], f_eqn, t_spec)
net = NeuralNetwork('models/CLF_controller_layer_num_3')
clsys = NNCSystem(sys, net, 'jacobian')
t_end = 3

x0 = np.array([
    np.interval(-0.45,-0.44),
    np.interval(-0.55,-0.54),
    np.interval(0.65,0.66),
    np.interval(-0.75,-0.74),
    np.interval(0.85,0.86),
    np.interval(-0.65,-0.64)
])
xcent, xpert = get_cent_pert(x0)
# print(net.seq[0].weight.detach().numpy() @ xcent + net.seq[0].bias.detach().numpy())

partitioner = UniformPartitioner(clsys)
popts = UniformPartitioner.Opts(0, 0)
# partitioner = CGPartitioner(clsys)
# popts = CGPartitioner.Opts(0.05, 0.2, 2, 0)

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
    plt.plot(traj(tt)[:,0], traj(tt)[:,1], color='tab:red', zorder=0)

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

