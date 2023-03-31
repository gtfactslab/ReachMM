import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from time import time
from torch.multiprocessing import Pool
from ReachMM.reach import Trajectory, Partition, width
from ReachMM import ControlFunction, ControlInclusionFunction
from ReachMM import DisturbanceFunction, DisturbanceInclusionFunction
from ReachMM import NoDisturbance, NoDisturbanceIF
from ReachMM.decomp import d_positive, d_metzler

class MixedMonotoneModel :
    def __init__ (self, control:ControlFunction=None, control_if:ControlInclusionFunction=None, u_step=None,
                        disturbance:DisturbanceFunction=None, disturbance_if:DisturbanceInclusionFunction=None) :

        if control is None and control_if is None :
            Exception("Need to define 'control' or 'control_if'")
        if u_step is None :
            Exception("Need to define u_step")
        if disturbance is None :
            disturbance = NoDisturbance()
        if disturbance_if is None :
            disturbance_if = NoDisturbanceIF()

        self.control = control
        self.control_if = control_if
        self.disturbance = disturbance
        self.disturbance_if = disturbance_if
        self.u_step = u_step
        self.embed = None
        self.sum_func = 0


    # Abstract Method for embedding system
    def f  (self, x, u, w=None) : 
        pass

    # Abstract Method for decomposition function
    def d  (self, x, xh, u, uh, w, wh) :
        pass

    def d_i (self, i, x, xh, u, uh, w, wh) :
        return self.d(x, xh, u, uh, w, wh)[i]

    def func_ (self, t, xt) :
        if not self.embed :
            return self.f(xt,self.control.u_calc,self.disturbance.w(t,xt))
        else :
            dim = len(xt) // 2
            x  = xt[:dim]; xh = xt[dim:]
            if self.control_if.mode == 'global':
                w  = self.disturbance_if.w (t,xt)
                wh = self.disturbance_if.wh(t,xt)
                xdot  = self.d(x, xh,\
                               self.control_if.u_calc,
                               self.control_if.uh_calc,
                               w, wh)
                xhdot = self.d(xh, x, \
                               self.control_if.uh_calc,
                               self.control_if.u_calc,
                               wh, w)
            elif self.control_if.mode == 'hybrid' or self.control_if.mode == 'local' :
                xdot  = [self.d_i(i,x,xh, \
                                  self.control_if.u_calc_x[i,:],\
                                  self.control_if.uh_calc_x[i,:],\
                                  self.disturbance_if.w_i (i,t,xt,False), \
                                  self.disturbance_if.wh_i(i,t,xt,False)) \
                                    for i in range(dim)]
                xhdot = [self.d_i(i,xh,x, \
                                  self.control_if.uh_calc_x[i,:],\
                                  self.control_if.u_calc_x[i,:],\
                                  self.disturbance_if.wh_i(i,t,xt,True), \
                                  self.disturbance_if.w_i (i,t,xt,True)) \
                                    for i in range(dim)]
                self.sum_func += 1
            elif self.control_if.mode == 'disclti' :
                xdot  = self.control_if._Mm @ x + \
                        self.control_if._Mn @ xh + \
                        self.control_if.Bp @ self.control_if._d.reshape(-1) + \
                        self.control_if.Bn @ self.control_if.d_.reshape(-1) + \
                        self.c
                xhdot = self.control_if.M_m @ xh + \
                        self.control_if.M_n @ x + \
                        self.control_if.Bn @ self.control_if._d.reshape(-1) + \
                        self.control_if.Bp @ self.control_if.d_.reshape(-1) + \
                        self.c
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
    
    def compute_trajectory (self, x0, t_span, method='RK45', t_step=None, enable_bar=True) :
        self.embed = False
        
        traj = Trajectory(x0,self,self.control,t_step=t_step)
        for t0 in tqdm(np.arange(t_span[0],t_span[1],self.u_step), disable=(not enable_bar)) :
            # print(f'traj.integrate([{t0}, {t0+self.u_step}])')
            traj.integrate([t0, t0+self.u_step], method)

        return traj

    def compute_reachable_set (self, x_xh0, t_span, control_divisions=0, integral_divisions=0, method='RK45', t_step=None, repartition=False, enable_bar=True) :
        print(f"compute_reachable_set: cd={control_divisions}, id={integral_divisions}")
        self.embed = True
        rs = Partition(x_xh0,self,self.control_if,True,
                       self.disturbance_if, t_step=t_step)
        for i in range(control_divisions) :
            rs.cut_all(True)
        for i in range(integral_divisions) :
            rs.cut_all(False)

        for t0 in tqdm(np.arange(t_span[0],t_span[1],self.u_step), disable=(not enable_bar)) :
            rs.integrate([t0, t0+self.u_step], method)
            if repartition :
                rs.repartition(t0+self.u_step)
                # rs.x_xh0 = rs(t0 + self.u_step)
                # rs.subpartitions = None
                # for i in range(control_divisions) :
                #     rs.cut_all(True, False, round(t0/t_step))
                # for i in range(integral_divisions) :
                #     rs.cut_all(False, False, round(t0/t_step))
        
        return rs

    def compute_reachable_set_eps (self, x_xh0, t_span, control_divisions=0, integral_divisions=0, method='RK45', t_step=None, eps=1, max_primer_depth=1, max_depth=2, check_contr=0.5, cut_dist=False, repartition=False, enable_bar=True, axs=None) -> Partition :
        print(f"compute_reachable_set_eps: cd={control_divisions}, id={integral_divisions}, eps={eps}, max_primer_depth={max_primer_depth}, max_depth={max_depth}, check_contr={check_contr}, cut_dist={cut_dist}, repartition={repartition}")
        self.embed = True
        rs = Partition(x_xh0,self,self.control_if,True,
                       self.disturbance_if, t_step=t_step)
        for i in range(control_divisions) :
            rs.cut_all(True,cut_dist)
        for i in range(integral_divisions) :
            rs.cut_all(False,cut_dist)

        for i, t0 in enumerate(tqdm(np.arange(t_span[0],t_span[1],self.u_step), disable=(not enable_bar))) :
            rs.integrate_eps([t0, t0+self.u_step], method, eps, max_primer_depth, max_depth, check_contr, cut_dist)
            if axs is not None :
                axs[i].clear()
                rs.draw_tree(axs[i],prog='dot')
                axs[i].set_title(f'$t_{i}$', x=0.25, y=0.925, fontsize=20)

            if repartition :
                rs.repartition(t0+self.u_step)

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
    