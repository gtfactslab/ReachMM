from ReachMM import MixedMonotoneModel, ControlFunction
from ReachMM import NoDisturbance, NoDisturbanceIF
from ReachMM.decomp import d_b1b2
import numpy as np
from casadi import *

weps = 0.1

class XYDoubleIntegratorModel (MixedMonotoneModel) :
    def __init__(self, control=None, control_if=None, u_step=0.25, disturbance=NoDisturbance(1), disturbance_if=NoDisturbanceIF(1)):
        super().__init__(control, control_if, u_step, disturbance, disturbance_if)

    def f(self, x, u, w) :
        return np.array([x[2],x[3],(1 + w[0])*u[0],(1 + w[0])*u[1]])

    def d(self, x, xh, u, uh, w, wh) :
        u0 = (1 + w[0])*u[0] if u[0] >= 0 else (1 + wh[0])*u[0]
        u1 = (1 + w[0])*u[1] if u[1] >= 0 else (1 + wh[0])*u[1]
        return np.array([x[2],x[3], u0, u1])

    def d_i (self, i, x, xh, u, uh, w, wh) :
        if i == 0 :
            return x[2]
        if i == 1 :
            return x[3]
        if i == 2 : 
            return (1 + w[0])*u[0] if u[0] >= 0 else (1 + wh[0])*u[0]
        return (1 + w[0])*u[1] if u[1] >= 0 else (1 + wh[0])*u[1]

class XYDoubleIntegratorMPC (ControlFunction):
    def __init__(self, n_horizon=20, u_step=2.5e-1, euler_steps=10):
        super().__init__(u_len=2)

        self.n_horizon = n_horizon
        self.u_step = u_step
        self.euler_steps = euler_steps

        x = MX.sym('x',4,1)
        px = x[0]; py = x[1]; vx = x[2]; vy = x[3]
        u = MX.sym('u',2,1)
        u1 = u[0]; u2 = u[1]

        xdot = MX(4,1)
        xdot[0] = vx
        xdot[1] = vy
        xdot[2] = u1
        xdot[3] = u2

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
            J += self.xx[0,n]**2 + self.xx[1,n]**2 + 0.5*self.uu[0,n]**2 + 0.5*self.uu[1,n]**2
            if(n > 0):
                J += 0.5*(self.uu[0,n] - self.uu[0,n-1])**2 + 0.5*(self.uu[1,n] - self.uu[1,n-1])
            J += 1e5*self.slack[0,n]**2
            # self.opti.subject_to(atan(100*fmin(self.xx[0,n] - 1.5,4.5 - self.xx[0,n])) + atan(100*fmin(self.xx[1,n] - 1.5,4.5 - self.xx[1,n])) <= 0 + self.slack[0,n])
            # self.opti.subject_to(atan(100*fmin(self.xx[0,n] + 4.5,-1.5 - self.xx[0,n])) + atan(100*fmin(self.xx[1,n] - 1.5,4.5 - self.xx[1,n])) <= 0 + self.slack[0,n])
            self.opti.subject_to((self.xx[0,n]-4)**2 + (self.xx[1,n]-4)**2 >= 3**2 - self.slack[0,n])
            self.opti.subject_to((self.xx[0,n]+4)**2 + (self.xx[1,n]-4)**2 >= 3**2 - self.slack[0,n])
            # self.opti.subject_to((self.xx[0,n]-4)**2 + (self.xx[1,n]-4)**2 - 3**2 >= 0 - self.slack[0,n])
            # self.opti.subject_to((self.xx[0,n]-4)**2 + (self.xx[1,n]+4)**2 >= 3**2 - self.slack[0,n])
            # self.opti.subject_to(((self.xx[0,n]+3)/2)**2 + ((self.xx[1,n])/3)**2 >= 1**2 - self.slack[0,n])
        J += 100*self.xx[0,N]**2 + 100*self.xx[1,N]**2
        self.opti.minimize(J)

        self.opti.subject_to(self.opti.bounded(-20,self.xx[0,:],20))
        self.opti.subject_to(self.opti.bounded(-20,self.xx[1,:],20))
        self.opti.subject_to(self.opti.bounded(-20,self.xx[2,:],20))
        self.opti.subject_to(self.opti.bounded(-20,self.xx[3,:],20))
        self.opti.subject_to(self.opti.bounded(-30,self.uu[0,:],30))
        self.opti.subject_to(self.opti.bounded(-30,self.uu[1,:],30))

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
