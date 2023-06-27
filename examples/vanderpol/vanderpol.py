import numpy as np
import interval
import sympy as sp
import matplotlib.pyplot as plt

from ReachMM import AutonomousSystem, AffineRefine
from ReachMM import ContinuousTimeSpec
from ReachMM.utils import plot_iarray_t

x1, x2, y1, y2 = (x_vars := sp.symbols('x1 x2 y1 y2'))

f_eqn = [
    x2,
    (1 - x1**2)*x2 - x1,
    x1*(x1*x2 + 1),
    (1 - x1**2)*x2 - y1
]

x0_1 = np.interval(1.399,1.400)
x0_2 = np.interval(2.299,2.300)

x0 = np.array([
    x0_1,
    x0_2,
    x0_1 - x0_2,
    x0_1 + x0_2
])

t_spec = ContinuousTimeSpec(0.1, 0.1)
ref = AffineRefine(
    M = np.array([
        [-1, 1, 1, 0],
        [-1,-1, 0, 1]
    ]), 
    b = np.array([0, 0])
)
sys = AutonomousSystem(x_vars, f_eqn, t_spec, ref)

from inspect import getsource
print(getsource(sys.sys.f))

t_span = [0, 10]
tt = t_spec.tt(*t_span)
traj = sys.compute_trajectory(*t_span, x0)
# print(traj(tt))

fig, axs = plt.subplots(1,2,dpi=100,figsize=[8,4],squeeze=False)
# fig.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.95, wspace=0.15, hspace=0.25)
fig.subplots_adjust(left=0.075, right=0.95, bottom=0.125, top=0.925, wspace=0.125, hspace=0.25)

plot_iarray_t(axs[0,0], tt, traj(tt)[:,0], color='tab:blue')
plot_iarray_t(axs[0,1], tt, traj(tt)[:,1], color='tab:blue')

axs[0,0].set_ylim([-5,5])
axs[0,1].set_ylim([-5,5])

plt.show()
