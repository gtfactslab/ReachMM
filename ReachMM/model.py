import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from ReachMM.reach import *
from time import time
from torch.multiprocessing import Pool

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
    
    def compute_trajectory (self, x0, t_span, t_step, method='RK45', enable_bar=True, embed=None) :
        if embed == None :
            embed = self.control == None
        self.embed = embed

        len_tt = 1 + int(self.u_step / t_step)
        tt_ustep = np.arange(t_span[0], t_span[1], self.u_step)
        length = (len_tt-1) * len(tt_ustep) + 1
        t = np.empty((length,))
        x = np.empty((length,len(x0)))
        if self.control.u_len == None :
            if not self.embed :
                self.control.u_len = len(self.control(0,x0))
            else :
                self.control.u_len = len(self.control_if(0,x0)[0])
        if not self.embed :
            u = np.empty((length,self.control.u_len))
            udisc = np.empty((len(tt_ustep),self.control.u_len))
        else :
            u = np.empty((length,2*self.control_if.u_len))
            udisc = np.empty((len(tt_ustep),2*self.control_if.u_len))
        t[0] = t_span[0]
        x[0,:] = x0

        timesum = 0

        for ind,section in enumerate(tqdm(tt_ustep, disable=(not enable_bar))) :
            tt = np.linspace(section, section+self.u_step, len_tt)

            before = time()
            if not self.embed :
                self.control.step(tt[0], x[ind*(len_tt-1)])
                udisc[ind,:] = self.control.u_calc
            else :
                self.control_if.prime(x[ind*(len_tt-1)])
                self.control_if.step(tt[0], x[ind*(len_tt-1)])
                # udisc[ind,:] = np.concatenate((self.control_if.u_calc, self.control_if.uh_calc))

            # sol = solve_ivp(self.func_, [tt[0], tt[-1]], x[ind*(len_tt-1),:], t_eval=tt, method=method)
            xx = self.integrate(x[ind*(len_tt-1),:], tt, method=method)
            after = time()
            # print(type(sol))
            # print(sol.t.shape, tt.shape, sol.y.shape)
            t[1+ind*(len_tt-1):1+(ind+1)*(len_tt-1)] = tt[1:]
            x[1+ind*(len_tt-1):1+(ind+1)*(len_tt-1),:] = xx[:,1:].T
            u[ind*(len_tt-1):(ind+1)*(len_tt-1),:] = np.ones((len_tt-1,1)) @ udisc[ind,:].reshape(1,-1)
            timesum += after - before
        
        # print(timesum)
        return {'t':t[0:-1], 'x':x[0:-1,:].T, 'u':u[1:,:].T, 'udisc': udisc}


    def compute_reachable_set (self, x_xh0, t_span, t_step, control_divisions=0, integral_divisions=0, repartition=True, method='RK45', enable_bar=True) :
        self.embed = True

        len_tt = 1 + int(self.u_step / t_step)
        tt_ustep = np.arange(t_span[0], t_span[1], self.u_step)
        
        rs = ReachableSet(len(tt_ustep) + 1)
        rs_calc = ReachableSet(len(tt_ustep) + 1)
        rs.add_control_partition(ControlPartition(x_xh=x_xh0),0)
        rs.create_partitions(control_divisions, integral_divisions, 0)

        self.sum_func = 0

        for i,section in enumerate(tqdm(tt_ustep, disable=(not enable_bar))) :
            # x_xh=rs.get_bounding_box(i)

            if repartition :
                rs_calc.add_control_partition(ControlPartition(x_xh=rs.get_bounding_box(i)),i)
                rs_calc.create_partitions(control_divisions, integral_divisions, i)
                for cp in (rs_calc.partitions_i[i]) :
                    tt = np.linspace(section,section+self.u_step,len_tt)
                    self.control_if.prime(cp.x_xh)
                    self.control_if.step(tt[0], cp.x_xh)
                    xx = self.integrate(cp.x_xh, tt, method=method)
                    newcp = ControlPartition(x_xh_t=xx[:,1:], tt=tt[1:])
                    if cp.integral_partitions is not None :
                        for ip in cp.integral_partitions :
                            self.control_if.step(tt[0],ip.x_xh)
                            xx = self.integrate(ip.x_xh, tt, method=method)
                            newcp.add_integral_partition(Partition(x_xh_t=xx[:,1:], tt=tt[1:]))
                    rs.add_control_partition(newcp,i+1)
            else :
                for cp in (rs.partitions_i[i]) :
                    tt = np.linspace(section,section+self.u_step,len_tt)
                    self.control_if.prime(cp.x_xh)
                    self.control_if.step(tt[0], cp.x_xh)
                    xx = self.integrate(cp.x_xh, tt, method=method)
                    newcp = ControlPartition(x_xh_t=xx[:,1:], tt=tt[1:])
                    if cp.integral_partitions is not None :
                        for ip in cp.integral_partitions :
                            self.control_if.step(tt[0],ip.x_xh)
                            xx = self.integrate(ip.x_xh, tt, method=method)
                            newcp.add_integral_partition(Partition(x_xh_t=xx[:,1:], tt=tt[1:]))
                        newcp.x_xh = newcp.get_bounding_box()
                        newcp.x_xh_t = newcp.get_bounding_box_t()
                    rs.add_control_partition(newcp,i+1)

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
