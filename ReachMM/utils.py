import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

def run_time (func, *args, **kwargs) :
    before = time.time()
    ret = func(*args, **kwargs)
    after = time.time()
    return ret, (after - before)

def uniform_disjoint (set, N) :
    probs = [s[1] - s[0] for s in set]; probs = probs / np.sum(probs)
    return np.array([np.random.choice([
        np.random.uniform(s[0], s[1]) for s in set
    ], p=probs) for _ in range(N)])

def gen_ics(RANGES, N) :
    X = np.empty((N, len(RANGES)))
    for i, range in enumerate(RANGES) :
        X[:,i] = uniform_disjoint(range, N)
    return X

def file_to_numpy (filenames) :
    with open('data/' + filenames[0] + '.npy', 'rb') as f :
        nploaded = np.load(f)
        X = nploaded['X']
        U = nploaded['U']
    for FILE in filenames[1:] :
        with open('data/' + FILE + '.npy', 'rb') as f :
            nploaded = np.load(f)
            X = np.append(X, nploaded['X'], axis=0)
            U = np.append(U, nploaded['U'], axis=0)
    return X,U

def numpy_to_file (X, U, filename) :
    with open(filename, 'wb') as f :
        np.savez(f, X=X, U=U)

def plot_Y_X (fig, ax, tt, XX, YY, xlim=[-15,15], ylim=[-15,15], show_colorbar=False, show_obs=True) :
    if show_obs :
        ax.add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon'))
        ax.add_patch(Circle((-4,4),3/1.25,lw=0,fc='salmon'))
    # ax.add_patch(Rectangle((-1,-1),2,2,lw=0,fc='darkgreen'))
    # ax.add_patch(Circle((0,0),1,lw=0,fc='g'))
    # ax.add_patch(Ellipse((-3,0),2*1.8,2*2.8,lw=0,fc='r'))
    # ax.add_patch(Rectangle((4-np.sqrt(2)*3/2,4-np.sqrt(2)*3/2),np.sqrt(2)*3,np.sqrt(2)*3))
    # ax.add_patch(Rectangle((-4-np.sqrt(2)*3/2,4-np.sqrt(2)*3/2),np.sqrt(2)*3,np.sqrt(2)*3))
    points = np.array([XX,YY]).T.reshape(-1,1,2)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    lc = LineCollection(segs, lw=2, cmap=plt.get_cmap('cividis'))
    lc.set_array(tt)
    ax.add_collection(lc)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel('$p_x$',labelpad=3); ax.set_ylabel('$p_y$',labelpad=3, rotation='horizontal')
    if show_colorbar :
        cb = fig.colorbar(lc, ax=ax, location='right', aspect=40, fraction=0.025, pad=0)
        cb.set_ticks([0,0.25,0.5,0.75,1,1.25])
        cb.set_label('t', rotation='horizontal')
        # cb.set_ticks(list(cb.get_ticks()) + [tt[-1]])
    ax.set_title("y vs x, color t")
