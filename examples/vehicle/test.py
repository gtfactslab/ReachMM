from Vehicle import *
import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from ReachMM import NeuralNetwork, NeuralNetworkControl
from ReachMM.utils import gen_ics
from ReachMM import DisturbanceFunction, DisturbanceInclusionFunction
from ReachMM import LinearControl, LinearControlIF
from control import lqr

A = np.array([ [0,0,1,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]])
B = np.array([ [0,0], [0,0], [1,0], [0,1] ])
Q = np.array([ [1,0,0,0], [0,1,0,0], [0,0,0,0], [0,0,0,0] ])
R = np.array([ [0.5,0], [0,0.5] ])

K, S, E = lqr(A, B, Q, R)

net = NeuralNetwork('models/100r100r2')
# control = VehicleMPCController()
control = NeuralNetworkControl(net)
# control = LinearControl(-K)
# model = VehicleModel(control)
model = LinVehicleModel(control)

t_end = 20*0.25
grain = 0.01

RANGES = [
    ([-10,10],),
    ([  5,10],),
    ([-10,10],),
    ([-10,10],),
]

X0 = gen_ics(RANGES, 100)

ind = 0
X0[ind,:] = np.array([5,8,0,0]); ind += 1

tt = np.arange(0,t_end+grain,grain)

for x0 in X0 :
    traj = model.compute_trajectory(x0=x0, enable_bar=True, t_span=[0,t_end], t_step=0.01, method='euler')
    xx = traj(tt)

    print(xx.T)

    print(traj(t_end))
    print(traj.u_disc)
    fig, axs = plt.subplots(1,3,dpi=100,figsize=[14,4])
    # fig, axs = plt.subplots(len(experiments),3+1,dpi=100,figsize=[14,4],squeeze=False)

    ax = axs[0]
    ax.add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon'))
    ax.add_patch(Circle((-4,4),3/1.25,lw=0,fc='salmon'))
    points = np.array([xx[0,:],xx[1,:]]).T.reshape(-1,1,2)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    lc = LineCollection(segs, lw=2, cmap=plt.get_cmap('cividis'))
    lc.set_array(tt)
    ax.add_collection(lc)
    ax.set_xlim([-10,10]); ax.set_ylim([-10,10])
    ax.set_xlabel('$p_x$',labelpad=3); ax.set_ylabel('$p_y$',labelpad=3, rotation='horizontal')
    cb = fig.colorbar(lc, ax=ax, location='right', aspect=40, fraction=0.025, pad=0)
    cb.set_ticks([0,0.25,0.5,0.75,1,1.25])
    cb.set_label('t', rotation='horizontal')
    # cb.set_ticks(list(cb.get_ticks()) + [tt[-1]])
    ax.set_title("y vs x, color t")

    ax = axs[1]
    ax.plot(tt, xx[0,:], label='p_x')
    ax.plot(tt, xx[1,:], label='p_y')
    ax.legend()

    ax = axs[2]
    ax.plot(tt, xx[2,:], label='v_x')
    ax.plot(tt, xx[3,:], label='v_y')
    # ax.plot(traj.sol.ts)
    ax.legend()

    plt.show()