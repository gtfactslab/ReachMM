import numpy as np
from numpy import tan, arctan2, arctan
from time import time
from ReachMM import MixedMonotoneModel, ControlFunction, ControlInclusionFunction
from ReachMM import DisturbanceFunction, DisturbanceInclusionFunction
from ReachMM.decomp import d_sin, d_cos, d_b1b2
from casadi import *

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
    def __init__(self, control: ControlFunction = None, control_if: ControlInclusionFunction = None, u_step=2.5e-1, 
                 lf=1, lr=1,
                 disturbance: DisturbanceFunction = None, disturbance_if: DisturbanceInclusionFunction = None):
        super().__init__(control, control_if, u_step, disturbance, disturbance_if)
        self.lf = lf
        self.lr = lr
        self.u_step = u_step

    def f(self, x, u, w) :
        u1, u2 = u
        X, Y, psi, v = x.ravel()
        beta = arctan2((self.lr * tan(u2)),(self.lf + self.lr))
        dX = v*np.cos(psi + beta)
        dY = v*np.sin(psi + beta)
        dpsi = v*np.sin(beta)/self.lr
        dv = u1
        xdot = np.array([dX, dY, dpsi, dv])
        return xdot
    
    def d(self, x, xh, u, uh, w, wh) :
        u1, u2 = u
        u1h, u2h = uh
        X, Y, psi, v = x.ravel()
        Xh, Yh, psih, vh = xh.ravel()
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
    
    def d_i(self, i, x, xh, u, uh, w, wh) :
        # u[0], u[1] = u
        if i == 3:
            return u[0]
        # u1h, uhat[1] = uhat
        # X, Y, x[2], x[3] = x
        # Xh, Yh, xhat[2], xhat[3] = xhat
        beta = arctan((self.lr * tan(u[1]))/(self.lf + self.lr))
        betah = arctan((self.lr * tan(uh[1]))/(self.lf + self.lr))
        if i == 0:
            return d_b1b2( (x[3] , d_cos(x[2] + beta, xh[2] + betah)), 
                        (xh[3], d_cos(xh[2] + betah, x[2] + beta)))
        if i == 1:
            return d_b1b2((x[3] , d_sin(x[2] + beta, xh[2] + betah)), 
                        (xh[3], d_sin(xh[2] + betah, x[2] + beta)))
        return d_b1b2( (x[3] , d_sin(beta, betah)),
                    (xh[3], d_sin(betah, beta)))

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
