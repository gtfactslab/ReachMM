import argparse

parser = argparse.ArgumentParser(description="Experiments for _____ Paper")

parser.add_argument('-T', '--t_end', help="End time for simulation", \
    type=float, default=None)
parser.add_argument('--integ_method', help="Integration Method",\
    choices=['RK45', 'RK23', 'euler'], default='RK45')
parser.add_argument('--lirpa_method', help="Bound Propogation Method for auto_LiRPA",\
    choices=['CROWN', 'CROWN-IBP', 'IBP'], default='CROWN')
parser.add_argument('--no_partitioning', help='Perform Partitioning Experiment',\
    action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--t_step', help="Time Step for Integration", \
    type=float, default=None)
parser.add_argument('-N','--runtime_N', help="Number of times to call reachable set estimator for time averaging",\
    type=int, default=1)

args = parser.parse_args()
print(args.lirpa_method)
print(args.t_end)
print(args.no_partitioning)

import numpy as np
from ReachMM import NeuralNetworkControl, NeuralNetworkControlIF
from VehicleNeuralNetwork import VehicleNeuralNetwork
from VehicleModel import VehicleModel
from VehicleUtils import *
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

nn = VehicleNeuralNetwork('twoobs-nost', device=device)
control = NeuralNetworkControl(nn, device=device)
controlif = NeuralNetworkControlIF(nn, mode='hybrid', method='CROWN', device=device)
model = VehicleModel(control, controlif, u_step=0.25)
x0 = np.array([8,8,-2*np.pi/3,2])
eps = np.array([0.1,0.1,0.01,0.01])
t_step = 0.01 if args.t_step is None else args.t_step
args.t_end = 1.25 if args.t_end is None else args.t_end
xlen = 4

t_span = [0,args.t_end]

x0d = np.concatenate((x0 - eps,x0 + eps))

traj, runtime = run_time(model.compute_trajectory, x0, t_span, t_step, method=args.integ_method, embed=False, enable_bar=False)
tt = traj['t']; xx = traj['x']; uu = traj['u']
print(f"One Real Trajectory Time: {runtime}")
dtraj, runtime = run_time(model.compute_trajectory, x0d, t_span, t_step, method=args.integ_method, embed=True, enable_bar=False)
dtt = dtraj['t']; dxx = dtraj['x']; duu = dtraj['u']
print(f"One Embedded Trajectory Time: {runtime}")

xname = ['$p_x$', '$p_y$', '$\psi$', '$v$']
pxx, pyy, ppsi, vv = xx
dpxx, dpyy, dppsi, dvv, dpxxh, dpyyh, dppsih, dvvh = dxx

# ====== Main Experiment ======
# experiments = [[('global',(1,0),'tab:blue',args.integ_method), ('hybrid',(1,0),'tab:orange',args.integ_method), ('local',(1,0),'tab:green',args.integ_method)]]
experiments = [[('hybrid',(0,1),'tab:orange',args.integ_method)]]
# experiments = [[('global',(1,0),'tab:blue',args.integ_method),('local',(1,0),'tab:green',args.integ_method),('hybrid',(1,0),'tab:orange',args.integ_method)]]
# fig, axs = plt.subplots(len(experiments),len(experiments[0])+1,dpi=100,figsize=[14,4],squeeze=False)
fig, axs = plt.subplots(len(experiments),3+1,dpi=100,figsize=[14,4],squeeze=False)
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.125, top=0.9, wspace=0.125, hspace=0.2)
# fig2, axs2 = plt.subplots(1,4,dpi=100,figsize=[12,3],squeeze=False)
# fig2.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95, wspace=0.3, hspace=0.2)
table = [['Mode, #CD / #ID', 'Runtime (s)']]

# Monte Carlo
MC_N = 100
x0mc = np.random.uniform(x0d[:xlen].reshape(-1,1)@np.ones((1,MC_N)),x0d[xlen:].reshape(-1,1)@np.ones((1,MC_N)),(xlen,MC_N)).T
mc = []
for n, x0mcn in enumerate(tqdm(x0mc)) :
    trajmc = model.compute_trajectory(x0mcn,t_span=[0,t_span[1]+model.u_step],t_step=model.u_step,embed=False,enable_bar=False,method=args.integ_method)
    mc.append(trajmc['x'])
    for i, l in enumerate(experiments):
        for j in range(len(l)):
            axs[i,j].scatter(trajmc['x'][0,:],trajmc['x'][1,:],s=0.01,c='g',alpha=0.1)

for i, l in enumerate(experiments) :
    for j, (mode, (cd, id), color, integ_method_pre) in enumerate(l) :
        controlif.mode = mode
        runtimes = np.empty(args.runtime_N)
        integ_method = integ_method_pre.split()
        for n in range(args.runtime_N) :
            before = time.time()
            rs = model.compute_epsilon_reachable_set(x0d, t_span, t_step if integ_method[0] != 'euler' else float(integ_method[1][1:-1]), \
                eps=1, control_divisions=cd, method=integ_method[0], enable_bar=False)
            # rs = model.compute_reachable_set(x0d, t_span, t_step if integ_method[0] != 'euler' else float(integ_method[1][1:-1]), \
            #     control_divisions=cd, integral_divisions=id, method=integ_method[0], enable_bar=False, repartition=True)
            after = time.time()
            runtimes[n] = after - before
        avg_runtime = runtimes.mean()
        std_runtime = runtimes.std()

        # print(f"Reachable Set Computation Time ({cd},{id}): {after - before}")
        plot_Y_X (fig,axs[i,j],tt,pxx,pyy,xlim=[-1,9],ylim=[-1,9])
        rs.draw_sg_boxes(axs[i,j])
        axs[i,j].set_title(f'$d^{mode[0].capitalize()}$, $D_a={(2**cd)**xlen}$, $D_s={(2**id)**xlen}$',fontdict=dict(fontsize=20))
        axs[i,j].text(-0.5,8.5,f'runtime:\n${avg_runtime:.3f}\pm{std_runtime:.3f}$',fontsize=15,verticalalignment='top')
        table.append([f'Reach-MM ({mode})', f'{avg_runtime:.3f}'])
        for k in [1] :
            axs[0,-1].plot(tt, xx[k,:], color='white', lw=0.5)
            rs.plot_bounds_t(axs[0,-1],k,color=color, label=f'$d^{mode[0].capitalize()}$')
            axs[0,-1].set_xlabel('t')
            axs[0,-1].set_ylabel(xname[k], rotation='horizontal')
            axs[0,-1].legend()

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


# ====== Partitioning Experiment ======
if not args.no_partitioning :
    experiments_part = [[(0,1), (1,0), (0,2), (1,1), (2,0)]]
    # experiments_part = [[(0,0), (0,1), (1,0)],\
    #                     [(0,2), (1,1), (2,0)]]

    fig_part, axs_part = plt.subplots(len(experiments_part),len(experiments_part[0]),dpi=100,figsize=[14,4],squeeze=False)
    fig_part.subplots_adjust(left=0.025, right=0.975, bottom=0.125, top=0.9, wspace=0.125, hspace=0.2)
    table = [['# Control / # Integration Divisions', 'Runtime (s)']]

    for i, l in enumerate(experiments_part) :
        for j, (cd, id) in enumerate(l) :
            controlif.mode = 'hybrid'
            runtimes = np.empty(args.runtime_N)
            integ_method = integ_method_pre.split()
            for n in range(args.runtime_N) :
                before = time.time()
                rs = model.compute_reachable_set(x0d, t_span, t_step if args.integ_method != 'euler' else model.u_step, \
                    control_divisions=cd, integral_divisions=id, method=args.integ_method, enable_bar=False, repartition=True)
                after = time.time()
                runtimes[n] = after - before
            avg_runtime = runtimes.mean()
            std_runtime = runtimes.std()

            # print(f"Reachable Set Computation Time ({cd},{id}): {after - before}")
            table.append([f'{cd} / {id}', f'{after - before:.3f}'])
            plot_Y_X (fig_part,axs_part[i,j],tt,pxx,pyy,xlim=[-1,9],ylim=[-1,9])
            rs.draw_sg_boxes(axs_part[i,j])
            axs_part[i,j].set_title(f'$d^H$, $D_a={(2**cd)**xlen}$, $D_s={(2**id)**xlen}$',fontdict=dict(fontsize=20))
            axs_part[i,j].text(-0.5,8.5,f'runtime:\n${avg_runtime:.3f}\pm{std_runtime:.3f}$',fontsize=15,verticalalignment='top')
            for point in mc :
                mcpxx, mcpyy, mcppsi, mcvv = point
                axs_part[i,j].scatter(mcpxx,mcpyy,s=0.01,c='g',alpha=0.1)

    print(tabulate(table, tablefmt='latex_raw'))
    print(tabulate(table))
    plt.savefig(f'figures/partitioning_experiment.pdf')

plt.show()