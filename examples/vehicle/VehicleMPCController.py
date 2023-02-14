import numpy as np
from ReachMM import ControlFunction
from casadi import *

NUM_TRAJS = 50000
FILENAME = 'twoobs'
PROCESSES = 8
PLOT_DATA = False

class VehicleMPCController (ControlFunction) :
    def __init__(self, n_horizon=20, u_step=2.5e-1, euler_steps=10, lr=1, lf=1):
        super().__init__(u_len=2)
        self.n_horizon = n_horizon
        self.u_step = u_step
        self.euler_steps = euler_steps

        x = MX.sym('x',4,1)
        X_cm = x[0]; Y_cm = x[1]; psi = x[2]; v = x[3]
        u = MX.sym('u',2,1)
        u1 = u[0]; u2 = u[1]

        xdot = MX(4,1)
        beta = arctan((lr * np.tan(u2))/(lf + lr))
        xdot[0] = v*cos(psi + beta)
        xdot[1] = v*sin(psi + beta)
        xdot[2] = v*sin(beta)/lr
        xdot[3] = u1

        f = Function('f',[x,u],[xdot],['x','u'],['xdot'])

        N = self.n_horizon

        # dae = {'x':x, 'p':u, 'ode':f(x,u)}
        # intg_opts = {'tf':self.t_step, 'simplify':True} #'number_of_finite_elements': 4
        # intg = integrator('intg','rk',dae,intg_opts)
        # res = intg(x0=x,p=u)
        # F = Function('F',[x,u],[res['xf']],['x','u'],['x_next'])

        def euler_integ (x,u) :
            step = self.u_step / self.euler_steps
            for t in np.arange(0,self.u_step,step) :
                x = x + step*f(x,u)
            return x
        euler_res = euler_integ(x,u)
        F = Function('F',[x,u],[euler_res],['x','u'],['x_next'])

        self.opti = Opti()
        self.xx = self.opti.variable(4,N+1)
        self.uu = self.opti.variable(2,N)
        self.x0 = self.opti.parameter(4,1)
        self.slack = self.opti.variable(1,N)

        self.opti.subject_to(self.xx[:,0] == self.x0)
        J = 0
        for n in range(N) :
            self.opti.subject_to(self.xx[:,n+1] == F(self.xx[:,n],self.uu[:,n]))
            J += self.xx[0,n]**2 + self.xx[1,n]**2 + 0.1*self.uu[0,n]**2 + 15*self.uu[1,n]**2
            if(n > 0):
                J += 5e-3*(self.uu[0,n] - self.uu[0,n-1])**2 + 5*(self.uu[1,n] - self.uu[1,n-1])
            J += 1e5*self.slack[0,n]**2
            # self.opti.subject_to(atan(100*fmin(self.xx[0,n] - 1.5,4.5 - self.xx[0,n])) + atan(100*fmin(self.xx[1,n] - 1.5,4.5 - self.xx[1,n])) <= 0 + self.slack[0,n])
            # self.opti.subject_to(atan(100*fmin(self.xx[0,n] + 4.5,-1.5 - self.xx[0,n])) + atan(100*fmin(self.xx[1,n] - 1.5,4.5 - self.xx[1,n])) <= 0 + self.slack[0,n])
            self.opti.subject_to((self.xx[0,n]-4)**2 + (self.xx[1,n]-4)**2 >= 3**2 - self.slack[0,n])
            self.opti.subject_to((self.xx[0,n]+4)**2 + (self.xx[1,n]-4)**2 >= 3**2 - self.slack[0,n])
            # self.opti.subject_to((self.xx[0,n]-4)**2 + (self.xx[1,n]-4)**2 - 3**2 >= 0 - self.slack[0,n])
            # self.opti.subject_to((self.xx[0,n]-4)**2 + (self.xx[1,n]+4)**2 >= 3**2 - self.slack[0,n])
            # self.opti.subject_to(((self.xx[0,n]+3)/2)**2 + ((self.xx[1,n])/3)**2 >= 1**2 - self.slack[0,n])
        J += 100*self.xx[0,N]**2 + 100*self.xx[1,N]**2 + 1*self.xx[3,N]**2
        self.opti.minimize(J)

        self.opti.subject_to(self.opti.bounded(-20,self.xx[0,:],20))
        self.opti.subject_to(self.opti.bounded(-20,self.xx[1,:],20))
        self.opti.subject_to(self.opti.bounded(-10,self.xx[3,:],10))
        self.opti.subject_to(self.opti.bounded(-20,self.uu[0,:],20))
        self.opti.subject_to(self.opti.bounded(-pi/3,self.uu[1,:],pi/3))

        # self.opti.solver('ipopt',{'print_time':0},{'linear_solver':'MA57', 'print_level':0, 'sb':'yes','max_iter':100000})
        self.opti.solver('ipopt',{'print_time':0},{'print_level':0, 'sb':'yes','max_iter':100000})
        # self.opti.solver('ipopt',{'print_time':0},{'print_level':0, 'sb':'yes','max_iter':100000})
        # self.opti.solver('ipopt',)

    def u(self, t, x):
        self.opti.set_value(self.x0, x)
        for n in range(self.n_horizon + 1) :
            self.opti.set_initial(self.xx[:,n], x)
        sol = self.opti.solve()
        # print(sol.value(self.slack))
        return sol.value(self.uu[:,0])

# if __name__ == '__main__' :
#     from tqdm import tqdm
#     from multiprocessing import Pool
#     from datetime import datetime
#     from matplotlib import pyplot as plt
#     from VehicleModel import VehicleModel
#     from VehicleUtils import gen_ics

#     FILEPATH = 'data/' + FILENAME + datetime.now().strftime('_%Y%m%d-%H%M%S') + '.npy'

#     problem_horizon = 20

#     NUM_POINTS = NUM_TRAJS * problem_horizon

#     print("Writing to " + FILEPATH)
#     print("TRAJECTORIES: %d" % NUM_TRAJS)
#     print("PROB HORIZON: %d" % problem_horizon)
#     print("TOTAL POINTS: %d" % NUM_POINTS)

#     XRANGE = ([-10,10],)
#     YRANGE = ([5,10],)
#     PRANGE = ([-np.pi,np.pi],)
#     VRANGE = ([-10,10],)

#     control = VehicleMPCController()
#     model = VehicleModel(u=control,u_step=control.u_step)

#     t_end = control.u_step * problem_horizon

#     X0 = gen_ics(XRANGE, YRANGE, PRANGE, VRANGE, NUM_TRAJS)

#     X = np.ones((NUM_POINTS, 4))
#     U = np.ones((NUM_POINTS, 2))

#     def task(x) :
#         try :
#             traj = model.compute_trajectory(x0=x, t_step=control.u_step, enable_bar=False, t_span=[0,t_end])
#             tt = traj['t']; xx = traj['x']; uu = traj['u']
#             return [tt, xx.T, uu.T]
#         except:
#             return task(gen_ics(XRANGE, YRANGE, PRANGE, VRANGE, 1)[0,:])

#     pool = Pool(processes=PROCESSES)

#     if PLOT_DATA :
#         fig, axs = plt.subplots(4,4)
#         axs = axs.reshape(-1)
#         axsi = 0

#     for i, result in enumerate(tqdm(pool.imap_unordered(task, X0), total=NUM_TRAJS, smoothing=0)) :
#         tt, xx, uu = result
#         X[i*problem_horizon:(i+1)*problem_horizon,:] = xx
#         U[i*problem_horizon:(i+1)*problem_horizon,:] = uu
#         if PLOT_DATA :
#             # axsi = (axsi + 1) % len(axs); ax = axs[axsi]; ax.clear()
#             # cmap = sns.cubehelix_palette(rot=-0.4, as_cmap=True)
#             # points = ax.scatter(xx[:,0], xx[:,1], c=tt, cmap=cmap, s=1)
#             # ax.set_xlim([-15,15]); ax.set_ylim([-15,15])
#             # # fig.colorbar(points, ax=ax)
#             # # ax.set_title("y vs x, color t")
#             # plt.ion(); plt.show(); plt.pause(0.00000001)
#             pass

#     with open(FILEPATH, 'wb') as f :
#         np.savez(f, X=X, U=U)

