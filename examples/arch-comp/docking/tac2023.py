import argparse
parser = argparse.ArgumentParser(description="Vehicle (bicycle model) Experiments for L4DC 2023 Paper")
parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",\
                    type=int, default=1)
args = parser.parse_args()

import numpy as np
import interval
from interval import from_cent_pert, get_lu, get_cent_pert
import sympy as sp
from ReachMM.time import *
from ReachMM.system import *
from ReachMM.reach import UniformPartitioner, CGPartitioner
from ReachMM.control import ConstantDisturbance
from ReachMM.utils import run_times, plot_iarray_t
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
spec1 = sp.sqrt(sxd**2 + syd**2)
spec2 = (0.2 + 2*n*sp.sqrt(sx**2 + sy**2))
print(spec)
spec_lam = sp.lambdify((x_vars,), spec, 'numpy')
spec1_lam = sp.lambdify((x_vars,), spec1, 'numpy')
spec2_lam = sp.lambdify((x_vars,), spec2, 'numpy')

t_spec = ContinuousTimeSpec(0.1,1)
sys = System(x_vars, u_vars, [w], f_eqn, t_spec)
net = NeuralNetwork('models/model')
# del(net.seq[-1])
# del(net.seq[-1])
clsys = NNCSystem(sys, net, NNCSystem.InclOpts('interconnect'))
print(clsys)
t_end = 40

def run_and_plot (ax, x0, ind) :
    xcent, xpert = get_cent_pert(x0)

    partitioner = UniformPartitioner(clsys)
    popts = UniformPartitioner.Opts(0,0)

    tt = t_spec.tt(0,t_end)

    def run () :
        rs = partitioner.compute_reachable_set(0,t_end,x0,popts)
        safe = rs.check_safety(spec_lam, tt)
        return rs, safe
    (rs, safe), times = run_times(args.runtime_N, run)

    print(f'Safe: {safe} in {np.mean(times)} \\pm {np.std(times)} (s)')

    print(rs(t_end))
    # plot_iarray_t(ax, tt, np.array([spec_lam(rs(t)) for t in tt]), color='tab:blue')
    plot_iarray_t(ax, tt, np.array([spec1_lam(rs(t)) for t in tt]), color='tab:red')
    plot_iarray_t(ax, tt, np.array([spec2_lam(rs(t)) for t in tt]), color='tab:blue')

    # ax.plot(tt,np.zeros_like(tt), color='tab:red')
    # title = f'${x0}$'.replace(') ', '\\times').replace('(','').replace('[[','[').replace(')]','')
    title = f'$\\mathcal{{X}}_0^{ind}={x0}$'.replace(') ', '\\times').replace('(','').replace('[[','[').replace(')]','')
    ax.set_title(title, fontsize=18)
    # ax.text(-0.075, 1.025, f'$\\mathcal{{X}}_0^{ind}$', transform=ax.transAxes)

    ax.set_xlabel('Time (s)', fontsize=20)

plt.rc('font', size=18)
# fig, axs = plt.subplots(2,2,dpi=100,figsize=[16,8],squeeze=False)
# fig.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.95, wspace=0.125, hspace=0.3)
fig, axs = plt.subplots(2,2,dpi=100,figsize=[16,8],squeeze=False)
fig.subplots_adjust(left=0.055, right=0.99, bottom=0.085, top=0.95, wspace=0.125, hspace=0.35)

# axs[0,0].set_ylabel('$g(s_x,s_y,\dot{{s}}_x,\dot{{s}}_y)$')
# axs[1,0].set_ylabel('$g(s_x,s_y,\dot{{s}}_x,\dot{{s}}_y)$')

# (0,0)
run_and_plot(axs[0,0], np.array([
    np.interval(102,106),
    np.interval(102,106),
    np.interval(-0.28,-0.24),
    np.interval(-0.28,-0.24)
]), 0)

# (0,1)
run_and_plot(axs[0,1], np.array([
    np.interval(102,106),
    np.interval(102,106),
    np.interval(0.24,0.28),
    np.interval(0.24,0.28)
]), 1)

# (1,0)
run_and_plot(axs[1,0], np.array([
    np.interval(70,74),
    np.interval(70,74),
    np.interval(-0.28,-0.24),
    np.interval(-0.28,-0.24)
]), 2)

# (1,1)
run_and_plot(axs[1,1], np.array([
    np.interval(70,74),
    np.interval(70,74),
    np.interval(0.24,0.28),
    np.interval(0.24,0.28)
]), 3)

fig.savefig('figures/docking_tac2023.pdf')
plt.show()
