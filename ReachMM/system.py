from __future__ import annotations
import numpy as np
import sympy as sp
import interval
from interval import get_lu, get_iarray, has_nan
from ReachMM.time import *
from ReachMM.neural import NeuralNetwork, NeuralNetworkControl
from ReachMM.control import Disturbance, NoDisturbance, Control
from ReachMM.utils import d_metzler, d_positive
from pprint import pformat
from numba import jit
from inspect import getsource

class Trajectory :
    def __init__(self, t_spec:TimeSpec, t0, x0, t_alloc=None) -> None:
        self.t_spec = t_spec
        self.t0 = t0
        self.tf = t0
        t_alloc = t0 + 10 if t_alloc is None else t_alloc
        self.xx = np.empty((self.t_spec.lentt(t0,t_alloc)+1,) + x0.shape, x0.dtype)

        self._n = lambda t : np.round((t - self.t0)/self.t_spec.t_step).astype(int)
        self.set(t0,x0)

    def set (self, t, x) :
        if self._n(t) > self._n(self.tf) :
            self.tf = t
        self.xx[self._n(t),:] = x

    def __call__(self, t) :
        not_def = np.logical_or(self._n(t) > self._n(self.tf), self._n(t) < self._n(self.t0))
        if np.any(not_def) :
            raise Exception(f'Trajectory not defined at {t[not_def]} \\notin [{self.t0},{self.tf}]')
        return self.xx[self._n(t),:]

class NLSystem :
    def __init__(self, x_vars, u_vars, w_vars, f_eqn, t_spec:TimeSpec) -> None:
        self.t_spec = t_spec
        self.x_vars = sp.Matrix(x_vars)
        self.u_vars = sp.Matrix(u_vars)
        self.w_vars = sp.Matrix(w_vars)

        if t_spec.type == 'discrete' or t_spec.type == 'continuous' :
            self.f_eqn  = sp.Matrix(f_eqn)
        elif t_spec.type == 'discretized' :
            self.f_eqn = self.x_vars + t_spec.t_step*sp.Matrix(f_eqn)

        def my_cse(exprs, symbols=None, optimizations=None, postprocess=None,
            order='canonical', ignore=(), list=True) :
            return sp.cse(exprs=exprs, symbols=sp.numbered_symbols('_dum'), optimizations='basic', 
                          postprocess=postprocess, order=order, ignore=ignore, list=list)

        tuple = (x_vars, u_vars, w_vars)

        self.f    = sp.lambdify(tuple, self.f_eqn, 'numpy', cse=my_cse)
        self.f_i  = [sp.lambdify(tuple, f_eqn_i, 'numpy', cse=my_cse) for f_eqn_i in self.f_eqn]
        self.f_len = len(self.f_i)

        print(self.f_eqn.jacobian(u_vars))
        self.Df_x = sp.lambdify(tuple, self.f_eqn.jacobian(x_vars), 'numpy', cse=my_cse)
        self.Df_u = sp.lambdify(tuple, self.f_eqn.jacobian(u_vars), 'numpy', cse=my_cse)
        self.Df_w = sp.lambdify(tuple, self.f_eqn.jacobian(w_vars), 'numpy', cse=my_cse)

    def __str__ (self) :
        return f'''Nonlinear {self.t_spec.__str__()} System with 
            \r  {'xdot' if self.t_spec.type == 'continuous' or self.t_spec.type == 'discretized' else 'x+'} = f(x,u,w) = {self.f_eqn.__str__()}'''

    def get_AB (self, x, u, w) :
        return self.Df_x(x, u, w)[0].astype(x.dtype), self.Df_u(x, u, w)[0].astype(x.dtype)

class ControlledSystem :
    def __init__(self, sys:NLSystem, control:Control, interc_mode=None,
                 dist:Disturbance=NoDisturbance(1)) :
        self.sys = sys
        self.control = control

        # Set default interconnection modes
        if interc_mode is None :
            if self.sys.t_spec.type == 'continuous' :
                interc_mode = 'hybrid'
            else :
                interc_mode = 'global'

        self.control.mode = interc_mode
        self.dist = dist
    
    # Returns x_{t+1} given x_t.
    def func (self, t, x) :
        self.control.step(t, x)

        # Monotone Inclusion
        if x.dtype == np.interval :
            if self.sys.t_spec.type == 'continuous' :
                # Interconnection mode by default
                return x + self.sys.t_spec.t_step*self.f_replace(x)
            else :
                return self.sys.f(x, self.control.iuCALC, self.dist.w(t,x))[0].reshape(-1)
        # Deterministic system
        else :
            if self.sys.t_spec.type == 'continuous' :
                return x + self.sys.t_spec.t_step*self.sys.f(x,self.control.uCALC,self.dist.w(t,x))[0].reshape(-1)
            else :
                return self.sys.f(x,self.control.uCALC,self.dist.w(t,x))[0].reshape(-1)

    # Abstract method to prepare for the next control interval.
    def prime (self, x) :
        # self.control.prime(x)
        pass

    def compute_trajectory (self, t0, tf, x0) :
        xx = Trajectory(self.sys.t_spec, t0, x0, tf)

        for tt in self.sys.t_spec.tu(t0, tf) :
            self.control.prime(xx(tt[0]))
            self.control.step(0,xx(tt[0]))
            self.prime(xx(tt[0]))
            for j, t in enumerate(tt) :
                xx.set(t + self.sys.t_spec.t_step, self.func(t, xx(t)))

        return xx
    
    def f_replace (self, x, iuCALC_x=None, iuCALCx_=None) :
        ret = np.empty_like(x)
        if iuCALC_x is None :
            iuCALC_x = self.control.iuCALC_x
        if iuCALCx_ is None :
            iuCALCx_ = self.control.iuCALCx_
        for i in range(len(x)) :
            xi = np.copy(x); xi[i].u = x[i].l
            _reti = self.sys.f_i[i] (xi, iuCALC_x[i,:], self.dist.w(0,xi))[0]
            xi[i].u = x[i].u; xi[i].l = x[i].u
            ret_i = self.sys.f_i[i] (xi, iuCALCx_[i,:], self.dist.w(0,xi))[0]
            ret[i] = np.intersection(_reti, ret_i)
        return ret

class AutonomousSystem (ControlledSystem) :
    pass

class NNCSystem (ControlledSystem) :
    def __init__(self, sys:NLSystem, nn:NeuralNetwork, incl_method='jacobian', interc_mode=None,
                 dist:Disturbance=NoDisturbance(1), uclip=np.interval(-np.inf,np.inf)) -> None:
        self.nn = nn
        ControlledSystem.__init__(self, sys, NeuralNetworkControl(nn, interc_mode, uclip=uclip),
                                  interc_mode, dist)
        self.incl_method = incl_method
        self.e = None
        self.uj = None
        self.ujCALCx_ = None
        self.ujCALC_x = None
    
    def prime (self, x):
        # self.control.prime(x)
        if self.incl_method == 'jacobian' :
            if self.sys.t_spec.type == 'continuous' :
                self.e = np.zeros(self.sys.f_len, dtype=np.interval)
                # self.control.step(0, x)
                self.uj = np.copy(self.control.iuCALC)
                self.ujCALCx_ = np.copy(self.control.iuCALCx_)
                self.ujCALC_x = np.copy(self.control.iuCALC_x)
    
    def func (self, t, x) :
        # Returns x_{t+1} given x_t (euler disc. for continuous time based on t_spec.t_step)
        # Assumes access to pre-computed control in self.control (call control.step before this)
        # Monotone Inclusion
        if x.dtype == np.interval :
            if self.incl_method == 'jacobian' :
                if self.sys.t_spec.type == 'continuous' :
                    # self.control.step(t, x)

                    _x, x_ = get_lu(x)
                    _u, u_ = get_lu(self.uj)
                    A, B = self.sys.get_AB(x, self.uj, self.dist.w(t,x))
                    _A, A_ = get_lu(A)
                    _B, B_ = get_lu(B)
                    _Bp, _Bn = d_positive(_B)
                    B_p, B_n = d_positive(B_)

                    _e, e_ = get_lu(self.e)

                    # Centered around _x, _u
                    _K = _Bp@self.control._C + _Bn@self.control.C_
                    K_ = B_p@self.control.C_ + B_n@self.control._C
                    _Kp, _Kn = d_positive(_K)
                    K_p, K_n = d_positive(K_)
                    _c = - _Kn@_e - _Kp@e_ 
                    c_ = - K_n@e_ - K_p@_e
                    # _c = 0
                    # c_ = 0
                    _L = _A + _K
                    L_ = A_ + K_
                    _Lp, _Ln = d_metzler(_L)
                    L_p, L_n = d_metzler(L_)
                    f = self.sys.f(_x,_u,self.dist.w(t,_x))[0].reshape(-1)
                    d_x1 = _Lp@_x + _Ln@x_ + _c - _A@_x - _B@_u + _Bp@self.control._d + _Bn@self.control.d_ + f
                    dx_1 = L_p@x_ + L_n@_x + c_ - A_@_x - B_@_u + B_p@self.control.d_ + B_n@self.control._d + f


                    # Centered around _x, u_
                    _K = B_p@self.control._C + B_n@self.control.C_
                    K_ = _Bp@self.control.C_ + _Bn@self.control._C
                    _Kp, _Kn = d_positive(_K)
                    K_p, K_n = d_positive(K_)
                    _c = - _Kn@_e - _Kp@e_ 
                    c_ = - K_n@e_ - K_p@_e
                    _L = _A + _K
                    L_ = A_ + K_
                    _Lp, _Ln = d_metzler(_L)
                    L_p, L_n = d_metzler(L_)
                    f = self.sys.f(_x,u_,self.dist.w(t,_x))[0].reshape(-1)
                    d_x2 = _Lp@_x + _Ln@x_ + _c - _A@_x - B_@u_ + B_p@self.control._d + B_n@self.control.d_ + f
                    dx_2 = L_p@x_ + L_n@_x + c_ - A_@_x - _B@u_ + _Bp@self.control.d_ + _Bn@self.control._d + f

                    # Centered around x_, _u
                    _K = _Bp@self.control._C + _Bn@self.control.C_
                    K_ = B_p@self.control.C_ + B_n@self.control._C
                    _Kp, _Kn = d_positive(_K)
                    K_p, K_n = d_positive(K_)
                    _c = - _Kn@_e - _Kp@e_ 
                    c_ = - K_n@e_ - K_p@_e
                    _L = A_ + _K
                    L_ = _A + K_
                    _Lp, _Ln = d_metzler(_L)
                    L_p, L_n = d_metzler(L_)
                    f = self.sys.f(x_,_u,self.dist.w(t,x_))[0].reshape(-1)
                    d_x3 = _Lp@_x + _Ln@x_ + _c - A_@x_ - _B@_u + _Bp@self.control._d + _Bn@self.control.d_ + f
                    dx_3 = L_p@x_ + L_n@_x + c_ - _A@x_ - B_@_u + B_p@self.control.d_ + B_n@self.control._d + f

                    # Centered around x_, u_
                    _K = B_p@self.control._C + B_n@self.control.C_
                    K_ = _Bp@self.control.C_ + _Bn@self.control._C
                    _Kp, _Kn = d_positive(_K)
                    K_p, K_n = d_positive(K_)
                    _c = - _Kn@_e - _Kp@e_ 
                    c_ = - K_n@e_ - K_p@_e
                    _L = A_ + _K
                    L_ = _A + K_
                    _Lp, _Ln = d_metzler(_L)
                    L_p, L_n = d_metzler(L_)
                    f = self.sys.f(x_,u_,self.dist.w(t,x_))[0].reshape(-1)
                    d_x4 = _Lp@_x + _Ln@x_ + _c - A_@x_ - B_@u_ + B_p@self.control._d + B_n@self.control.d_ + f
                    dx_4 = L_p@x_ + L_n@_x + c_ - _A@x_ - _B@u_ + _Bp@self.control.d_ + _Bn@self.control._d + f

                    # Interconnection mode
                    dx5 = self.f_replace(x, self.ujCALC_x, self.ujCALCx_)
                    d_x5, dx_5 = get_lu(dx5)

                    # Bounding the difference: error dynamics
                    self.control.step(0, x)
                    self.e = self.e + self.sys.t_spec.t_step * self.sys.f(x, self.control.iuCALC, self.dist.w(t,x))[0].reshape(-1)

                    # _xtp1 = _x + self.sys.t_spec.t_step * np.max(np.array([d_x1,d_x4]), axis=0)
                    # x_tp1 = x_ + self.sys.t_spec.t_step * np.min(np.array([dx_1,dx_4]), axis=0)
                    # _xtp1 = _x + self.sys.t_spec.t_step * d_x5
                    # x_tp1 = x_ + self.sys.t_spec.t_step * dx_5
                    _xtp1 = _x + self.sys.t_spec.t_step * np.max(np.array([d_x1,d_x2,d_x3,d_x4,d_x5]), axis=0)
                    x_tp1 = x_ + self.sys.t_spec.t_step * np.min(np.array([dx_1,dx_2,dx_3,dx_4,dx_5]), axis=0)
                    return get_iarray(_xtp1, x_tp1)
                    # return np.intersection(x + self.sys.t_spec.t_step * get_iarray(d_x1, dx_1),
                    #                        x + self.sys.t_spec.t_step * get_iarray(d_x2, dx_2))
                    # return x + self.sys.t_spec.t_step * get_iarray(d_x1, dx_1)
                else :
                    _x, x_ = get_lu(x)
                    _u, u_ = get_lu(self.control.iuCALC)
                    A, B = self.sys.get_AB(x, self.control.iuCALC, self.dist.w(t,x))

                    _A, A_ = get_lu(A)
                    _B, B_ = get_lu(B)
                    _Bp, _Bn = d_positive(_B)
                    B_p, B_n = d_positive(B_)
                    
                    # Centered around _x
                    _L = _A + _Bp@self.control._C + _Bn@self.control.C_
                    L_ = A_ + B_p@self.control.C_ + B_n@self.control._C
                    _Lp, _Ln = d_positive(_L)
                    L_p, L_n = d_positive(L_)
                    f = self.sys.f(_x,_u,self.dist.w(t,_x))[0].reshape(-1)
                    d_x1 = _Lp@_x + _Ln@x_ - _A@_x - _B@_u + _Bp@self.control._d + _Bn@self.control.d_ + f
                    dx_1 = L_p@x_ + L_n@_x - A_@_x - B_@_u + B_p@self.control.d_ + B_n@self.control._d + f

                    # Centered around x_
                    _L = A_ + B_p@self.control._C + B_n@self.control.C_
                    L_ = _A + _Bp@self.control.C_ + _Bn@self.control._C
                    _Lp, _Ln = d_positive(_L)
                    L_p, L_n = d_positive(L_)
                    f = self.sys.f(x_,u_,self.dist.w(t,x_))[0].reshape(-1)
                    d_x4 = _Lp@_x + _Ln@x_ - A_@x_ - B_@u_ + B_p@self.control._d + B_n@self.control.d_ + f
                    dx_4 = L_p@x_ + L_n@_x - _A@x_ - _B@u_ + _Bp@self.control.d_ + _Bn@self.control._d + f
                    return np.intersection(get_iarray(d_x1, dx_1),get_iarray(d_x4, dx_4))

            elif self.incl_method == 'interconnect' :
                if self.sys.t_spec.type == 'continuous' :
                    return x + self.sys.t_spec.t_step*self.f_replace(x)
                else :
                    # print(x, self.control.iuCALC, self.dist.w(t,x))
                    return self.sys.f(x, self.control.iuCALC, self.dist.w(t,x))[0].reshape(-1)
        # Deterministic system
        else :
            if self.sys.t_spec.type == 'continuous' :
                return x + self.sys.t_spec.t_step*self.sys.f(x,self.control.uCALC,self.dist.w(t,x))[0].reshape(-1)
            else :
                return self.sys.f(x,self.control.uCALC,self.dist.w(t,x))[0].reshape(-1)

    def __str__ (self) :
        return f'''===== Closed Loop System Definition =====
            \r{self.sys.__str__()}
            \rcontrolled by {self.control.__str__()}
            \rusing the {self.incl_method} monotone inclusion function'''

if __name__ == '__main__' :
    px, py, psi, v, u1, u2, w = sp.symbols('p_x p_y psi v u1 u2 w')
    beta = sp.atan(sp.tan(u2)/2)
    f_eqn = [
        v*sp.cos(psi + beta), 
        v*sp.sin(psi + beta), 
        v*sp.sin(beta),
        u1
    ]

    uclip = np.array([
        np.interval(-20,20),
        np.interval(-np.pi/4,np.pi/4)
    ])

    # t_spec = DiscretizedTimeSpec(0.1)
    t_spec = ContinuousTimeSpec(0.05,0.25)
    sys = NLSystem([px, py, psi, v], [u1, u2], [w], f_eqn, t_spec)
    net = NeuralNetwork('../examples/vehicle/models/100r100r2')
    clsys = NNCSystem(sys, net, 'jacobian', uclip=uclip)
    # clsys = NNCSystem(sys, net, 'interconnect', uclip=uclip)

    t_span = [0,1]
    # tt = t_spec.tt(0,1)
    # print(t_spec.tu(0,2))

    from interval import from_cent_pert
    cent = np.array([8,8,-2*np.pi/3,2])
    pert = np.array([0.1,0.1,0.01,0.01])
    # pert = np.array([0.001,0.001,0.001,0.001])
    # pert = np.array([0.01,0.01,0.01,0.01])
    x0 = from_cent_pert(cent, pert)

    from ReachMM.utils import run_times, draw_iarrays

    xx, times = run_times(10, clsys.compute_trajectory, t_span[0],t_span[1],x0)
    print(np.mean(times), '\pm', np.std(times))
    
    tt = t_spec.tt(t_span[0],t_span[1])
    rs = xx(tt)

    # print(xx(np.arange(0,1+0.1,0.1)))
    # print(xx(t_spec.tt(t_span[0], t_span[1])))
    # print(xx(t_spec.tt(0,1)))

    import matplotlib.pyplot as plt

    traj = clsys.compute_trajectory(t_span[0], t_span[1], cent)
    plt.plot(traj(tt)[:,0], traj(tt)[:,1])
    plt.scatter(traj(tt)[:,0], traj(tt)[:,1],s=5)
    draw_iarrays(plt, rs)
    plt.show()
