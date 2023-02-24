import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from time import time
from torch.multiprocessing import Pool
from ReachMM.reach import Partition

class MixedMonotoneModel :
    def __init__ (self, control=None, control_if=None, u_step=0.1) :
        if control == None and control_if == None :
            Exception("Need to define 'control' or 'control_if'")

        self.control = control
        self.control_if = control_if
        self.u_step = u_step
        self.embed = None
        self.sum_func = 0

    # Abstract Method for embedding system
    def f  (self, x, u) : 
        pass

    # Abstract Method for decomposition function
    def d  (self, x, u, xh, uh) :
        pass

    def d_i (self, i, x, u, xh, uh) :
        return self.d(x, u, xh, uh)[i]

    def func_ (self, t, xt) :
        if not self.embed :
            return self.f(xt,self.control.u_calc)
        else :
            dim = len(xt) // 2
            x  = xt[:dim]; xh = xt[dim:]
            if self.control_if.mode == 'global' :
                xdot  = self.d(x,self.control_if.u_calc,xh,self.control_if.uh_calc)
                xhdot = self.d(xh,self.control_if.uh_calc,x,self.control_if.u_calc)
            elif self.control_if.mode == 'hybrid' or self.control_if.mode == 'local' :
                xdot = [self.d_i(i,x, self.control_if.u_calc_x[i,:], xh,self.control_if.uh_calc_x[i,:]) for i in range(dim)]
                xhdot = [self.d_i(i,xh,self.control_if.uh_calc_xh[i,:],x,self.control_if.u_calc_xh[i,:]) for i in range(dim)]
                self.sum_func += 1
            return np.concatenate((xdot,xhdot))
    
    def S (self, x0, x1) :
        n = len(x0)
        S = np.empty((n,n))
        u = self.control.u(0,x0)
        f0 = self.f(x0, u)
        for i in range(n):
            x = np.copy(x0)
            x[i] = x1[i]
            S[i,:] = self.f(x,u) - f0
        return S
    
    # def compute_trajectory (self, x0, t_span, method='RK45', t_step=None, enable_bar=True) :
    #     self.embed = False
    #     if t_step is None and method == 'euler' :
    #         Exception(f'Calling {method} method without t_step')
    #     elif t_step is not None and method != 'euler' :
    #         Exception(f'Calling {method} method with t_step')
        
    #     sol = None
    #     for t0 in tqdm(np.arange(t_span[0],t_span[1],self.u_step), disable=(not enable_bar)) :
    #         if method == 'euler':
    #             if sol is None :
    #                 sol = [self.x_xh0]
    #             for n in range(int(t_span[0]/self.t_step),int(t_span[1]/self.t_step)):
    #                 self.sol.append(self.sol[n] + self.t_step*self.func(n*self.t_step, self.sol[n]))
    #         else :


    def compute_reachable_set (self, x_xh0, t_span, control_divisions=0, integral_divisions=0, method='RK45', t_step=None, enable_bar=True) :
        self.embed = True
        rs = Partition(x_xh0,self.func_,self.control_if,True,t_step=t_step)
        for i in range(control_divisions) :
            rs.cut_all(True)
        for i in range(integral_divisions) :
            rs.cut_all(False)

        for t0 in tqdm(np.arange(t_span[0],t_span[1],self.u_step), disable=(not enable_bar)) :
            rs.integrate([t0, t0+self.u_step], method)
        
        return rs

    def integrate (self, x0, t_eval, method='RK45') :
        if method == 'euler' :
            dt = t_eval[1] - t_eval[0]
            xx = np.empty((len(x0), len(t_eval)))
            xx[:,0] = x0
            for i in range(0,len(t_eval)-1) :
                xdot = self.func_(t_eval[i], xx[:,i])
                xx[:,i+1] = xx[:,i] + dt*xdot
            return xx
        else :
            t_span = [t_eval[0], t_eval[-1]]
            sol = solve_ivp(self.func_, t_span, x0, method, t_eval)
            return sol.y
