import argparse

parser = argparse.ArgumentParser(description="Experiments for _____ Paper")

parser.add_argument('-T', '--t_end', help="End time for simulation", \
    type=float, default=None)
parser.add_argument('--integ_method', help="Integration Method",\
    choices=['RK45', 'RK23', 'euler'], default='RK45')
parser.add_argument('--lirpa_method', help="Bound Propogation Method for auto_LiRPA",\
    choices=['CROWN', 'CROWN-IBP', 'IBP'], default='CROWN')
parser.add_argument('--t_step', help="Time Step for Integration", \
    type=float, default=None)
parser.add_argument('-N','--runtime_N', help="Number of times to call reachable set estimator for time averaging",\
    type=int, default=1)

args = parser.parse_args()
print(args.lirpa_method)
print(args.t_end)

import numpy as np
from ReachMM import NeuralNetworkControl, NeuralNetworkControlIF
from QuadrotorModel import QuadrotorModel
from QuadrotorNeuralNetwork import QuadrotorNeuralNetwork
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from tabulate import tabulate

def run_time (func, *args, **kwargs) :
    before = time.time()
    ret = func(*args, **kwargs)
    after = time.time()
    return ret, (after - before)

device = 'cpu'

# nn = load_controller(system='Quadrotor',model_name='default').to(device)
nn = QuadrotorNeuralNetwork(device=device)
control = NeuralNetworkControl(nn, device=device)
controlif = NeuralNetworkControlIF(nn, mode='hybrid', method='CROWN', device=device)
model = QuadrotorModel(control, controlif, u_step=0.1)
x0 = np.array([4.7,4.7,3,0.95,0,0])
eps = np.array([0.05,0.05,0.05,0.01,0.01,0.01])
t_step = 0.01 if args.t_step is None else args.t_step
args.t_end = 1.2 if args.t_end is None else args.t_end
xlen = 6

t_span = [0,args.t_end]

x0d = np.concatenate((x0 - eps,x0 + eps))

traj, runtime = run_time(model.compute_trajectory, x0, t_span, t_step, method=args.integ_method, embed=False, enable_bar=False)
tt = traj['t']; xx = traj['x']; uu = traj['u']
print(f"One Real Trajectory Time: {runtime}")
dtraj, runtime = run_time(model.compute_trajectory, x0d, t_span, t_step, method=args.integ_method, embed=True, enable_bar=False)
dtt = dtraj['t']; dxx = dtraj['x']; duu = dtraj['u']
print(f"One Embedded Trajectory Time: {runtime}")

xname = ['$p_x$', '$p_y$', '$p_z$', '4', '5', '6']
pxx, pyy, pzz, vvx, vvy, vvz = xx
dpxx, dpyy, dpzz, dvvx, dvvy, dvvz, dpxxh, dpyyh, dpzzh, dvvxh, dvvyh, dvvzh = dxx

# ====== Main Experiment ======
experiments = [[('hybrid',(0,0),'tab:blue','RK45'), ('hybrid',(0,0),'tab:orange',f'euler ({t_step})'), ('hybrid',(0,0),'tab:orange',f'euler ({model.u_step})')]]
fig, axs = plt.subplots(len(experiments),len(experiments[0]),dpi=100,figsize=[14,5],squeeze=False,subplot_kw=dict(projection='3d'))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
# fig2, axs2 = plt.subplots(1,6,dpi=100,figsize=[12,3],squeeze=False)
# fig2.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
table = [['Method',f'Average Runtime (s) {args.runtime_N} calls','Reachable Set Estimate']]

# Monte Carlo
MC_N = 100
x0mc = np.random.uniform(x0d[:xlen].reshape(-1,1)@np.ones((1,MC_N)),x0d[xlen:].reshape(-1,1)@np.ones((1,MC_N)),(xlen,MC_N)).T
mc = []
for n, x0mcn in enumerate(tqdm(x0mc)) :
    for i, l in enumerate(experiments):
        for j,(mode, (cd, id), color, integ_method_pre) in enumerate(l) :
            integ_method = integ_method_pre.split()
            trajmc = model.compute_trajectory(x0mcn,t_span=[0,t_span[1]+model.u_step], t_step=model.u_step if integ_method[0] != 'euler' else float(integ_method[1][1:-1]),embed=False,enable_bar=False,method=integ_method[0])
            n = 1 if integ_method[0] != 'euler' else round(model.u_step / float(integ_method[1][1:-1]))
            axs[i,j].scatter3D(trajmc['x'][0,::n],trajmc['x'][1,::n],trajmc['x'][2,::n],s=1,c='g',alpha=1)

for i, l in enumerate(experiments) :
    for j, (mode, (cd, id), color, integ_method_pre) in enumerate(l) :
        controlif.mode = mode
        runtimes = np.empty(args.runtime_N)
        integ_method = integ_method_pre.split()
        for n in range(args.runtime_N) :
            before = time.time()
            rs = model.compute_reachable_set(x0d, t_span, t_step if integ_method[0] != 'euler' else float(integ_method[1][1:-1]), \
                control_divisions=cd, integral_divisions=id, method=integ_method[0], enable_bar=False)
            after = time.time()
            runtimes[n] = after - before
        avg_runtime = runtimes.mean()
        std_runtime = runtimes.std()

        # print(f"Reachable Set Computation Time ({cd},{id}): {after - before}")
        if integ_method[0] != 'euler' :
            axs[i,j].plot(pxx, pyy, pzz)
        rs.draw_3d_boxes(axs[i,j])
        axs[i,j].set_title(f'integration: {integ_method_pre}',fontdict=dict(fontsize=20))
        axs[i,j].text2D(0.5,1,f'runtime: ${avg_runtime:.3f}\pm{std_runtime:.3f}$',fontsize=15,verticalalignment='top',horizontalalignment='center',transform=axs[i,j].transAxes)
        axs[i,j].set_xlim([4.4,5.1]); axs[i,j].set_ylim([3.4,4.9]); axs[i,j].set_zlim([-3.5,3.5])
        axs[i,j].xaxis.set_rotate_label(False); axs[i,j].set_xlabel('$p_x$')
        axs[i,j].yaxis.set_rotate_label(False); axs[i,j].set_ylabel('$p_y$')
        axs[i,j].zaxis.set_rotate_label(False); axs[i,j].set_zlabel('$p_z$')
        table.append([f'Reach-MM ({mode}), {integ_method[0]}', f'runtime: ${avg_runtime:.3f}\pm{std_runtime:.3f}$', 0])

        # for k in range(xlen) :
        #     axs2[0,k].plot(tt, xx[k,:], color='white', lw=0.5)
        #     rs.plot_bounds_t(axs2[0,k],k,color=color, label=f'$d^{mode[0].capitalize()}$')
        #     axs2[0,k].set_xlabel('t')
        #     axs2[0,k].set_ylabel(xname[k], rotation='horizontal')
        #     axs2[0,k].legend()
# print(rs.get_bounding_box_t(-1))

print(tabulate(table, tablefmt='latex_raw'))
print(tabulate(table))


plt.figure(fig); plt.savefig(f'figures/main_experiment.pdf')
# plt.figure(fig2); plt.savefig(f'figures/main_experiment2.pdf')


plt.show()