import numpy as np
from numpy import tan, arctan2, arctan
from time import time
from ReachMM import MixedMonotoneModel
from ReachMM.decomp import d_sin, d_cos, d_b1b2

# For Controller Testing, run this file to simulate trajectories.
METHOD = 'nn'
NNFILE = 'twoobs'

'''
x = [
    X_cm, (center of mass x)
    Y_cm, (center of mass y)
    psi,  (heading angle)
    v,    (speed)
]
u = [
    u1, (acceleration),
    u2, (front steering angle)
]
'''

class VehicleModel (MixedMonotoneModel):
    def __init__(self, control=None, control_if=None, u_step=2.5e-1, lf=1, lr=1):
        super().__init__(control, control_if, u_step)
        self.lf = lf
        self.lr = lr
        self.u_step = u_step

    def f(self, x, u) :
        u1, u2 = u
        X, Y, psi, v = x.ravel()
        beta = arctan2((self.lr * tan(u2)),(self.lf + self.lr))
        dX = v*np.cos(psi + beta)
        dY = v*np.sin(psi + beta)
        dpsi = v*np.sin(beta)/self.lr
        dv = u1
        xdot = np.array([dX, dY, dpsi, dv])
        return xdot
    
    def d(self, x, u, xhat, uhat) :
        u1, u2 = u
        u1h, u2h = uhat
        X, Y, psi, v = x.ravel()
        Xh, Yh, psih, vh = xhat.ravel()
        beta = arctan2((self.lr * tan(u2)),(self.lf + self.lr))
        betah = arctan2((self.lr * tan(u2h)),(self.lf + self.lr))
        dX = d_b1b2( [v , d_cos(psi + beta, psih + betah)], 
                     [vh, d_cos(psih + betah, psi + beta)])
        dY = d_b1b2( [v , d_sin(psi + beta, psih + betah)], 
                     [vh, d_sin(psih + betah, psi + beta)])
        dpsi = d_b1b2( [v , d_sin(beta, betah)],
                       [vh, d_sin(betah, beta)])
        dv = u1
        xdot = np.array([dX, dY, dpsi, dv])
        return xdot
    
    def d_i(self, i, x, u, xhat, uhat) :
        # u[0], u[1] = u
        if i == 3:
            return u[0]
        # u1h, uhat[1] = uhat
        # X, Y, x[2], x[3] = x
        # Xh, Yh, xhat[2], xhat[3] = xhat
        beta = arctan((self.lr * tan(u[1]))/(self.lf + self.lr))
        betah = arctan((self.lr * tan(uhat[1]))/(self.lf + self.lr))
        if i == 0:
            return d_b1b2( (x[3] , d_cos(x[2] + beta, xhat[2] + betah)), 
                        (xhat[3], d_cos(xhat[2] + betah, x[2] + beta)))
        if i == 1:
            return d_b1b2((x[3] , d_sin(x[2] + beta, xhat[2] + betah)), 
                        (xhat[3], d_sin(xhat[2] + betah, x[2] + beta)))
        return d_b1b2( (x[3] , d_sin(beta, betah)),
                    (xhat[3], d_sin(betah, beta)))

# if __name__ == '__main__' :
#     from VehicleMPCController import VehicleMPCController
#     from VehicleUtils import *
#     from VehicleModel import VehicleModel
#     from VehicleNeuralNetwork import VehicleNeuralNetwork, VehicleStateTransformation
#     from NeuralNetworkControl import NeuralNetworkControl
#     import matplotlib.pyplot as plt
#     import time

#     XRANGE = ([-10,10],)
#     YRANGE = ([6,10],)
#     PRANGE = ([-np.pi,np.pi],)
#     VRANGE = ([-10,10],)

#     problem_horizon = 20
#     NUM_ICS = 100
#     t_step = 0.01

#     # control = VehicleMPCController()
#     nn = VehicleNeuralNetwork(file='twoobs-nost', device='cpu')
#     st = VehicleStateTransformation()
#     control = NeuralNetworkControl(nn)
#     model = VehicleModel(control)

#     t_span = [0,model.u_step*problem_horizon]

#     X = gen_ics(XRANGE, YRANGE, PRANGE, VRANGE, NUM_ICS)

#     ind = 0
#     X[ind,:] = np.array([7.5,7.5,-np.pi,2])
#     # X[ind,:] = np.array([ 5.64985442, 11.58482758, -0.7477311 , -1.68895928]); ind+=1
#     # X[ind,:] = np.array([-7.24177948,  9.12419163 ,-0.3415519 ,  4.44351421]); ind+=1
#     # X[ind,:] = np.array([ 5.345843 ,   8.58482883 ,-1.04108991, -3.87954384]); ind+=1
#     #[3.43104413 8.3884244  1.20541383 3.65459651]
#     # X[ind,:] = np.array([ 8.85002768,  9.92996661 ,-2.31952152, 0.80199218]); ind+=1
#     # X[ind,:] = np.array([ 12, 5, -1.57925116 , 0.44655594]); ind+=1
#     # X[ind,:] = np.array([ 8.26461452, 10.07690833, -1.57925116 , 0.44655594]); ind+=1
#     # X[ind,:] = np.array([ 5.2909565,  -8.05621893,  0.14837165, -2.62037427]); ind+=1
#     # X[ind,:] = np.array([-10, 0, 0, 0]); ind+=1493
#     for n, x0 in enumerate(X):
#         print(f'IC: {x0}')

#         fig, axs = plt.subplots(2,2,figsize=[8,8], dpi=100)
#         fig.set_tight_layout(True)
#         fig.suptitle("x0: " + np.array2string(x0))
        
#         before = time.time()
#         traj = model.compute_trajectory(x0=x0, t_span=t_span, t_step=t_step)
#         tt = traj['t']; xx = traj['x']; uu = traj['u']
#         after = time.time()
#         print(f'{after - before:.3f} seconds')
#         plot_solution(fig, axs, tt, xx, uu)

#         plt.show()