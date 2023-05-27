import argparse

from ReachMM.control import ConstantDisturbance
from ReachMM.time import ContinuousTimeSpec

parser = argparse.ArgumentParser(description="Vehicle (bicycle model) Experiments for L4DC 2023 Paper")
parser.add_argument('-N', '--runtime_N', help="Number of calls for time averaging",\
                    type=int, default=1)
args = parser.parse_args()

import numpy as np
import interval
from interval import from_cent_pert
import sympy as sp

from ReachMM import DiscretizedTimeSpec
from ReachMM import System, NeuralNetwork, NNCSystem
from ReachMM import UniformPartitioner
from ReachMM.utils import run_times, draw_iarrays_3d
import matplotlib.pyplot as plt

g = 9.81
px, py, pz, vx, vy, vz, u1, u2, u3, w = sp.symbols('p_x p_y p_z v_x v_y v_z u_1 u_2 u_3 w')
f_eqn = [
    vx,
    vy,
    vz,
    g*u1,
    -g*u2,
    u3 - g
]

uclip = np.array([
    np.interval(-np.pi/9,np.pi/9),
    np.interval(-np.pi/9,np.pi/9),
    np.interval(0,2*g)
])

# t_spec = ContinuousTimeSpec(0.1,0.1)
t_spec = DiscretizedTimeSpec(0.1)
sys = System([px, py, pz, vx, vy, vz], [u1, u2, u3], [w], f_eqn, t_spec)
net = NeuralNetwork('models/6r32r32r3-LP')
dist = ConstantDisturbance(np.array([g]),np.array([np.interval(g,g)]))
# clsys = NNCSystem(sys, net, 'jacobian', uclip=uclip, dist=dist)
clsys = NNCSystem(sys, net, 'jacobian', dist=dist)
# clsys = NNCSystem(sys, net, 'interconnect', uclip=uclip,dist=dist)
# clsys = NNCSystem(sys, net, 'interconnect', dist=dist)

print(clsys)

t_span = [0,1.2]

cent = np.array([4.7,4.7,3,0.95,0,0])
pert = np.array([0.05,0.05,0.05,0.01,0.01,0.01])
x0 = from_cent_pert(cent, pert)

rs, times = run_times(args.runtime_N, clsys.compute_trajectory, t_span[0], t_span[1], x0)
avg_runtime, std_runtime = np.mean(times), np.std(times)

print(avg_runtime, '\pm', std_runtime)
tt = t_spec.tt(t_span[0],t_span[1])
print(rs(tt))

fig, axs = plt.subplots(1,1,dpi=100,figsize=[5,5],squeeze=True,subplot_kw=dict(projection='3d'))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)

draw_iarrays_3d(axs, rs(tt))
axs.set_title(f'integration: euler (0.1)',fontdict=dict(fontsize=20))
axs.text2D(0.5,1,f'runtime: ${avg_runtime:.3f}\pm{std_runtime:.3f}$',fontsize=15,verticalalignment='top',horizontalalignment='center',transform=axs.transAxes)
axs.set_xlim([4.4,5.1]); axs.set_ylim([3.4,4.9]); axs.set_zlim([-3.5,3.5])
axs.xaxis.set_rotate_label(False); axs.set_xlabel('$p_x$')
axs.yaxis.set_rotate_label(False); axs.set_ylabel('$p_y$')
axs.zaxis.set_rotate_label(False); axs.set_zlabel('$p_z$')

mctrajs = clsys.compute_mc_trajectories(*t_span, x0, 20)
for mctraj in mctrajs :
    mctraj.scatter3d(axs, tt, s=0.25, c='r')

# traj = clsys.compute_trajectory(t_span[0], t_span[1], cent)
# xx = traj(tt)
# axs.scatter(xx[:,0], xx[:,1], xx[:,2])

# print(traj(t_span[1]))

# rs.draw_rs(plt, t_spec.uu(t_span[0], t_span[1]), color='r')

plt.show()