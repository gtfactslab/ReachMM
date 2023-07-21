import numpy as np
import interval
from interval import get_lu, get_cent_pert
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from ReachMM import AutonomousSystem, AffineRefine
from ReachMM import ContinuousTimeSpec
from ReachMM.utils import plot_iarray_t

from tqdm import tqdm

r, th = (x_vars := sp.symbols('r th'))

f_eqn = [
    -2*r,
    -10
]

r0 = np.interval(-1,-0.6)
th0 = np.interval(-np.pi/15,np.pi/15)
x0 = np.array([r0, th0])

def draw_angint (ax:plt.Axes, r, th, **kwargs) :
    thdeg = th * 180 / np.pi
    return ax.add_patch(Wedge((0,0), r.u, thdeg.l, thdeg.u, width=(r.u - r.l), **kwargs))

t_spec = ContinuousTimeSpec(0.01, 0.01)
plot_t_spec = ContinuousTimeSpec(0.1, 0.1)
sys = AutonomousSystem(x_vars, f_eqn, t_spec)
t_span = [0,2]
tt = t_spec.tt(*t_span)
traj = sys.compute_trajectory(*t_span, x0)
# mc_trajs = sys.compute_mc_trajectories(*t_span, x0, 10)

c = 0.05
mc_trajs = [sys.compute_trajectory(*t_span, xx) for xx in [
                # get_cent_pert(x0)[0], 
                # np.array([r0.l, th0.l]), np.array([r0.u, th0.u])
                np.array([r0.l+c, th0.u-c]), np.array([r0.u-c, th0.l+c]) 
            ]]

fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=100)

for t in plot_t_spec.tt(*t_span) :
    rt, tht = traj(t)
    draw_angint(ax, rt, tht, fc='none', ec='black', lw=2)

for mc_traj in mc_trajs :
    xx = mc_traj(tt)
    ax.plot(xx[:,0]*np.cos(xx[:,1]), xx[:,0]*np.sin(xx[:,1]), color='black')

ax.set_xlim((-1.5,1.5))
ax.set_ylim((-1.5,1.5))

plt.show()
