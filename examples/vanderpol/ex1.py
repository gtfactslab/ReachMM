import numpy as np
import interval
import sympy as sp
import matplotlib.pyplot as plt

from ReachMM import AutonomousSystem, AffineRefine
from ReachMM import ContinuousTimeSpec
from ReachMM.utils import plot_iarray_t

x1, x2, x3, y = (x_vars := sp.symbols('x1 x2 x3 y'))

f_eqn = [
    -x1*x2 + x3,
    2*x1*x2,
    y,
    x1*x2 + x3
]

i = np.interval(0,1)

x0 = np.array([
    i,
    i,
    i,
    i + i
])

t_spec = ContinuousTimeSpec(0.01, 0.01)
ref = AffineRefine(
    M = np.array([
        [-1,-1, 0, 1]
    ]), 
    b = np.array([0])
)
sys = AutonomousSystem(x_vars, f_eqn, t_spec, ref)

t_span = [0, 1]
tt = t_spec.tt(*t_span)
traj = sys.compute_trajectory(*t_span, x0)
print(traj(tt))

fig, axs = plt.subplots(1,2,dpi=100,figsize=[8,4],squeeze=False)
# fig.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.95, wspace=0.15, hspace=0.25)
fig.subplots_adjust(left=0.075, right=0.95, bottom=0.125, top=0.925, wspace=0.125, hspace=0.25)

plot_iarray_t(axs[0,0], tt, traj(tt)[:,2], color='tab:blue')
plot_iarray_t(axs[0,1], tt, traj(tt)[:,0], color='tab:blue')

axs[0,0].set_ylim([-5,30])
axs[0,1].set_ylim([-5,30])

plt.show()

