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
from matplotlib.patches import Rectangle, Wedge
import torch


t = sp.symbols('t')

x1, x2, x3, x4 = [sym(t) for sym in sp.symbols('x1 x2 x3 x4', cls=sp.Function)]
u, w = sp.symbols('u w')

x1.fdiff = lambda i : x2
x2.fdiff = lambda i : -x1 + 0.1*sp.sin(x3)
x3.fdiff = lambda i : x4
# x4.fdiff = lambda i : 11*sp.tanh(u)
x4.fdiff = lambda i : u - 10

r, th = [sym(t) for sym in sp.symbols('r th', cls=sp.Function)]
subs = [(x1, r*sp.cos(th)),(x2, r*sp.sin(th))]
r.fdiff  = lambda i : sp.simplify((x1**2 + x2**2).diff(t).subs(subs)/(2*r))
th.fdiff = lambda i : sp.simplify((sp.atan2(x2, x1).diff(t).subs(subs)))

x_vars = sp.Matrix([r, th, x3, x4])
f_eqn = list(x_vars.diff(t))
print(f_eqn)

def for_trans (x) :
    return np.array([np.sqrt(x[0]**2 + x[1]**2), np.arctan(x[1]/x[0]), x[2], x[3]])
def inv_trans (x) :
    return np.array([x[0]*np.cos(x[1]), x[0]*np.sin(x[1]), x[2], x[3]])

inv_trans_sym = [r*sp.cos(th), r*sp.sin(th), x3, x4]

# t_spec = ContinuousTimeSpec(0.05,0.5)
t_spec = ContinuousTimeSpec(0.1,0.1)
sys = System(x_vars, [u], [w], f_eqn, t_spec)
# net = NeuralNetwork('models/nn_tora_relu_tanh')
net = NeuralNetwork('models/controllerTora')
# Remove final tanh and offset layer.
# del(net.seq[-1])
# del(net.seq[-1])
print(net.seq)
clsys = NNCSystem(sys, net, incl_opts=NNCSystem.InclOpts('interconnect'),
                  g_tuple=(x_vars,), g_eqn=inv_trans_sym)
# clsys.set_standard_ordering()
# clsys.set_four_corners()
t_span = [0,0.1]
tt = t_spec.tt(*t_span)

x0 = np.array([
    np.interval(0,2),
    np.interval(-np.pi,np.pi),
    np.interval(-1,1),
    np.interval(-1,1)
])

# x0 = for_trans(np.array([
#     np.interval(-0.77,-0.75),
#     np.interval(-0.45,-0.43),
#     np.interval(0.51,0.54),
#     np.interval(-0.3,-0.28)
# ]))
# x0 = for_trans(np.array([
#     np.interval(0.6,0.7),
#     np.interval(-0.7,-0.6),
#     np.interval(-0.4,-0.3),
#     np.interval(0.5,0.6)
# ]))

partitioner = UniformPartitioner(clsys)
popts = UniformPartitioner.Opts(0,0)
# partitioner = CGPartitioner(clsys)
# popts = CGPartitioner.Opts(0.5, 2, 1)
rs, times = run_times(1,partitioner.compute_reachable_set,*t_span, x0, popts)
print(f'{np.mean(times)} \\pm {np.std(times)}')

fig, axs = plt.subplots(1,2,figsize=[8,4])

axs[0].add_patch(Rectangle((-0.1,-0.9), 0.3, 0.3, color='tab:green', alpha=0.5))

for t in tt :
    # xx = inv_trans(rs(t))
    # draw_iarray(axs[0], xx, 0, 1)
    xx = rs(t)
    axs[0].add_patch(Wedge((0,0), xx[0].u, xx[1].l*180/np.pi, xx[1].u*180/np.pi, xx[0].u - xx[0].l, lw=2, fill=False, color='tab:blue'))
    draw_iarray(axs[1], xx, 2, 3)

mc_trajs = clsys.compute_mc_trajectories(*t_span, x0, 100)
for mc_traj in mc_trajs :
    xx = inv_trans(mc_traj(tt).T)
    axs[0].plot(xx[0,:], xx[1,:], zorder=0)
    axs[1].plot(xx[2,:], xx[3,:], zorder=0)
    
plt.show()

