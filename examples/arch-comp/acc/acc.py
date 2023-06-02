import argparse
parser = argparse.ArgumentParser(description="ARCH-COMP Adaptive Cruise Control")
parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",
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
from ReachMM.utils import run_times, draw_iarray
import matplotlib.pyplot as plt

# States 
xlead, vlead, glead, xego, vego, gego = sp.symbols('xlead vlead glead xego vego gego')
x_vars = [xlead, vlead, glead, xego, vego, gego]
# Controls
aego = sp.symbols('aego')
# Disturbances
alead = sp.symbols('alead')
# Constants
u = 0.0001

f_eqn = [
    vlead,
    glead,
    -2*glead + 2*alead - u*vlead**2,
    vego,
    gego,
    -2*gego + 2*aego - u*vego**2
]
g_eqn = [
    vset := 30.0,
    Tgap := 1.4,
    vego,
    Drel := xlead - xego,
    vrel := vlead - vego
]
spec = (Drel - (Dsafe := (Ddefault:=10) + Tgap*vego))
print(spec)
spec_lam = sp.lambdify((x_vars,), spec, 'numpy')

t_spec = ContinuousTimeSpec(0.01,0.1)
sys = System(x_vars, [aego], [alead], f_eqn, t_spec)
net = NeuralNetwork('models/controller_5_20')
clsys = NNCSystem(sys, net, 'interconnect', 
                  dist=ConstantDisturbance([-2],[np.interval(-2,-2)]),
                  g_tuple=(x_vars,), g_eqn=g_eqn)
t_end = 5

x0 = np.array([
    np.interval(90,110),
    np.interval(32,32.2),
    np.interval(0,0),
    np.interval(10,11),
    np.interval(30,30.2),
    np.interval(0,0)
])
xcent, xpert = get_cent_pert(x0)

partitioner = UniformPartitioner(clsys)
popts = UniformPartitioner.Opts(0, 0)

tt = t_spec.tt(0,t_end)

def run () :
    rs = partitioner.compute_reachable_set(0,t_end,x0,popts)
    safe = rs.check_safety(spec_lam, tt)
    return rs, safe
(rs, safe), times = run_times(args.runtime_N, run)

print(f'Safe: {safe} in {np.mean(times)} \\pm {np.std(times)} (s)')

xx = rs(tt).T

plt.rc('font', size=14)
fig, ax = plt.subplots(1, 1, figsize=[6,4], squeeze=True)
fig.subplots_adjust(left=0.125, right=0.95, bottom=0.15, top=0.925, wspace=0.125, hspace=0.25)

Drel_xx  = sp.lambdify((x_vars,), Drel , 'numpy')(xx)
Drel_l, Drel_u = get_lu(Drel_xx)
Dsafe_xx = sp.lambdify((x_vars,), Dsafe, 'numpy')(xx)
Dsafe_l, Dsafe_u = get_lu(Dsafe_xx)

pltl = ax.plot(tt, Drel_l, color='tab:blue')
pltu = ax.plot(tt, Drel_u, color='tab:blue')
ax.fill_between(tt, Drel_l, Drel_u, color='tab:blue', alpha=0.25)
pltl = ax.plot(tt, Dsafe_l, color='tab:red')
pltu = ax.plot(tt, Dsafe_u, color='tab:red')
ax.fill_between(tt, Dsafe_l, Dsafe_u, color='tab:red', alpha=0.25)

ax.set_xlabel('Time (s)',labelpad=0.1)
ax.set_ylabel('Distance (m)',labelpad=0.1)

fig.savefig('figures/acc.pdf')

plt.show()
