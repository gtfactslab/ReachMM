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
specs = [ (x1 + 0.1), (0.2 - x1), (x2 + 0.9), (-0.6 - x2) ]
spec_lam = [sp.lambdify((x_vars,), spec, 'numpy') for spec in specs]

t_spec = ContinuousTimeSpec(0.005,0.5)
ref = AffineRefine(
    M = np.array([
        [-1, -1, 0, 0, 1, 0],
        [-1,  1, 0, 0, 0, 1],
    ]),
    b = np.array([ 0,0 ])
)
sys = System(x_vars, [u], [w], f_eqn, t_spec, ref)
net = NeuralNetwork('models/nn_tora_relu_tanh')
# Remove final tanh and offset layer.
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
clsys = NNCSystem(sys, net, incl_opts=NNCSystem.InclOpts('jacobian+interconnect'))
clsys.set_standard_ordering()
clsys.set_four_corners()
t_end = 5

x0 = np.array([
    np.interval(-0.77,-0.75),
    np.interval(-0.45,-0.43),
    np.interval(0.51,0.54),
    np.interval(-0.3,-0.28),
    np.interval(-0.77,-0.75) + np.interval(-0.45,-0.43),
    np.interval(-0.77,-0.75) - np.interval(-0.45,-0.43),
])
xcent, xpert = get_cent_pert(x0)

partitioner = UniformPartitioner(clsys)
popts = UniformPartitioner.Opts(0, 0)

tt = t_spec.tt(0,t_end)

def run () :
    rs = partitioner.compute_reachable_set(0,t_end,x0,popts)
    safe = rs.check_safety_tt(spec_lam, tt)
    print(safe)
    return rs, safe
(rs, safe), times = run_times(1, run)

print(f'Safe: {safe} in {np.mean(times)} \\pm {np.std(times)} (s)')

fig, axs = plt.subplots(1,2,figsize=[16,8],dpi=100,squeeze=False)
fig.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.925, wspace=0.125, hspace=0.25)

axs[0,0].add_patch(Rectangle((-0.1,-0.9), 0.3, 0.3, color='tab:green', alpha=0.5))
axs[0,1].add_patch(Rectangle((-0.1,-0.9), 0.3, 0.3, color='tab:green', alpha=0.5))

xx = rs(tt)
print(rs(t_end))
rs.draw_rs(axs[0,0], tt[::10])
rs.draw_rs(axs[0,1], tt)

trajs = clsys.compute_mc_trajectories(0,t_end,x0,100)
tt = clsys.sys.t_spec.tt(0,t_end)
for traj in trajs :
    for ax in axs.reshape(-1) :
        ax.plot(traj(tt)[:,0], traj(tt)[:,1], color='tab:red')
        ax.set_xlabel(f'$x_1$')
        ax.set_ylabel(f'$x_2$')

axs[0,1].set_xlim([-0.1,0.2])
axs[0,1].set_ylim([-0.9,-0.6])

# ax.set_xlabel(f'$x_1$')
# ax.set_ylabel(f'$x_2$')

fig.savefig('figures/tora_tac2023.pdf')
plt.show()

