from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime
from matplotlib import pyplot as plt
from XYDoubleIntegrator import *
from ReachMM.utils import gen_ics, numpy_to_file
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

NUM_TRAJS = 20000
FILENAME = 'twoobs'
PROCESSES = 8
PLOT_DATA = False

FILEPATH = 'data/' + FILENAME + datetime.now().strftime('_%Y%m%d-%H%M%S') + '.npy'

problem_horizon = 20

NUM_POINTS = NUM_TRAJS * problem_horizon

print("Writing to " + FILEPATH)
print("TRAJECTORIES: %d" % NUM_TRAJS)
print("PROB HORIZON: %d" % problem_horizon)
print("TOTAL POINTS: %d" % NUM_POINTS)

RANGES = [
    ([-10,10],),
    ([  5,10],),
    ([-10,10],),
    ([-10,10],),
]

control = XYDoubleIntegratorMPC()
model = XYDoubleIntegratorModel(control)

t_end = control.u_step * problem_horizon

X0 = gen_ics(RANGES, NUM_TRAJS)

X = np.ones((NUM_POINTS, 4))
U = np.ones((NUM_POINTS, 2))

tt = np.arange(0,t_end,control.u_step)

def task(x) :
    try :
        traj = model.compute_trajectory(x0=x, enable_bar=False, t_span=[0,t_end], t_step=0.01, method='euler')
        return traj(tt).T, traj.u_disc
    except Exception as e:
        print('failed... trying with new IC')
        print(e)
        return task(gen_ics(RANGES, 1)[0,:])

pool = Pool(processes=PROCESSES)

if PLOT_DATA :
    fig, axs = plt.subplots(4,4,dpi=100,figsize=[10,10])
    fig.set_tight_layout(True)
    axs = axs.reshape(-1)
    axsi = 0

for i, result in enumerate(tqdm(pool.imap_unordered(task, X0), total=NUM_TRAJS, smoothing=0)) :
    # tt, xx, uu = result
    # traj = result
    xx, uu = result

    X[i*problem_horizon:(i+1)*problem_horizon,:] = xx
    U[i*problem_horizon:(i+1)*problem_horizon,:] = uu
    if PLOT_DATA :
        axsi = (axsi + 1) % len(axs); ax = axs[axsi]; ax.clear()
        # ax = axs[0]
        ax.add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon'))
        ax.add_patch(Circle((-4,4),3/1.25,lw=0,fc='salmon'))
        # points = np.array([xx[:,0],xx[:,1]]).T.reshape(-1,1,2)
        # segs = np.concatenate([points[:-1],points[1:]],axis=1)
        # lc = LineCollection(segs, lw=2, cmap=plt.get_cmap('cividis'))
        # lc.set_array(tt)
        # ax.add_collection(lc)
        # ax.set_xlim([-10,10]); ax.set_ylim([-10,10])
        # ax.set_xlabel('$p_x$',labelpad=3); ax.set_ylabel('$p_y$',labelpad=3, rotation='horizontal')

        # cmap = sns.cubehelix_palette(rot=-0.4, as_cmap=True)
        points = ax.scatter(xx[:,0], xx[:,1], c=tt, cmap=plt.get_cmap('cividis'), s=1)
        ax.set_xlim([-10,10]); ax.set_ylim([-10,10])
        # ax.set_xlim([-15,15]); ax.set_ylim([-15,15])
        # fig.colorbar(points, ax=ax)
        # ax.set_title("y vs x, color t")
        plt.ion(); plt.show(); plt.pause(0.0001)
        pass

numpy_to_file(X, U, FILEPATH)
