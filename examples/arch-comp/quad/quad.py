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
from inspect import getsource

# States
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12')
x_vars = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12]
# Control
u1, u2, u3 = sp.symbols('u1 u2 u3')
u_vars = [u1, u2, u3]
# Disturbance
w_dist = sp.symbols('w_dist')

g = 9.81; m = 1.4
Jx = 0.054; Jy = 0.054; Jz = 0.104
tau = 0
f_eqn = [
    sp.cos(x8)*sp.cos(x9)*x4 + (sp.sin(x7)*sp.sin(x8)*sp.cos(x9) - sp.cos(x7)*sp.sin(x9))*x5 + (sp.cos(x7)*sp.sin(x8)*sp.cos(x9) + sp.sin(x7)*sp.sin(x9))*x6 ,
    sp.cos(x8)*sp.sin(x9)*x4 + (sp.sin(x7)*sp.sin(x8)*sp.sin(x9) - sp.cos(x7)*sp.cos(x9))*x5 + (sp.cos(x7)*sp.sin(x8)*sp.sin(x9) + sp.sin(x7)*sp.cos(x9))*x6,
    sp.sin(x8)*x4 - sp.sin(x7)*sp.cos(x8)*x5 - sp.cos(x7)*sp.cos(x8)*x6,
    x12*x5 * x11*x6 - g*sp.sin(x8),
    x10*x6 - x12*x4 + g*sp.cos(x8)*sp.sin(x7),
    x11*x4 - x10*x5 + g*sp.cos(x8)*sp.cos(x7) - g - u1/m,
    x10 + sp.sin(x7)*sp.tan(x8)*x11 + sp.cos(x7)*sp.tan(x8)*x12,
    sp.cos(x7)*x11 - sp.sin(x7)*x12,
    sp.sin(x7)*x11/sp.cos(x8) - sp.cos(x7)*x12/sp.cos(x8),
    (Jy - Jz)*x11*x12/Jx + u2/Jx,
    (Jz - Jx)*x10*x12/Jy + u3/Jy,
    (Jx - Jy)*x10*x11/Jz + tau/Jz
]
# spec = (Drel - (Dsafe := (Ddefault:=10) + Tgap*vego))
# print(spec)
# spec_lam = sp.lambdify((x_vars,), spec, 'numpy')

t_spec = ContinuousTimeSpec(0.025,0.1)
# t_spec = DiscretizedTimeSpec(0.1)
sys = System(x_vars, u_vars, [w_dist], f_eqn, t_spec)
print(sys)
net = NeuralNetwork('models/quad_controller_3_64')
clsys = NNCSystem(sys, net, 'jacobian')
t_end = 1

# print(getsource(sys.f_i[11]))

x0 = np.array([
    np.interval(-0.4,0.4),
    np.interval(-0.4,0.4),
    np.interval(-0.4,0.4),
    np.interval(-0.4,0.4),
    np.interval(-0.4,0.4),
    np.interval(-0.4,0.4),
    np.interval(0,0),
    np.interval(0,0),
    np.interval(0,0),
    np.interval(0,0),
    np.interval(0,0),
    np.interval(0,0)
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

print(rs(t_end))

xx = rs(tt)

fig, ax = plt.subplots(1, 1, squeeze=True)

x4l, x4u = get_lu(xx[:,2])

pltl = ax.plot(tt, x4l, color='tab:blue')
pltu = ax.plot(tt, x4u, color='tab:blue')
ax.fill_between(tt, x4l, x4u, color='tab:blue', alpha=0.25)

t_end = 5
tt = t_spec.tt(0,t_end)
trajs = clsys.compute_mc_trajectories(0,t_end,x0,100)
tt = clsys.sys.t_spec.tt(0,t_end)
for traj in trajs :
    plt.plot(tt, traj(tt)[:,2], color='tab:red')

plt.show()

