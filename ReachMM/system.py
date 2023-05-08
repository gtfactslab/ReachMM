from __future__ import annotations
import numpy as np
import sympy as sp
import interval
from interval import get_lu, get_iarray
from ReachMM.time import *
from ReachMM.neural import NeuralNetwork, NeuralNetworkControl
from ReachMM.control import Disturbance, NoDisturbance, Control
from ReachMM.reach import Trajectory, Partition
from ReachMM.utils import d_metzler, d_positive
from pprint import pformat
from numba import jit
from inspect import getsource

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

        self.Df_x = sp.lambdify(tuple, self.f_eqn.jacobian(x_vars), 'numpy', cse=my_cse)
        self.Df_u = sp.lambdify(tuple, self.f_eqn.jacobian(u_vars), 'numpy', cse=my_cse)
        self.Df_w = sp.lambdify(tuple, self.f_eqn.jacobian(w_vars), 'numpy', cse=my_cse)

    def __str__ (self) :
        return f'''Nonlinear {self.t_spec.__str__()} System with 
            \r  {'xdot' if self.t_spec.type == 'continuous' or self.t_spec.type == 'discretized' else 'x+'} = f(x,u,w) = {self.f_eqn.__str__()}'''

    def get_AB (self, x, u, w) :
        return self.Df_x(x, u, w)[0], self.Df_u(x, u, w)[0]

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
    
    # Abstract method returning x_{t+1} given x_t.
    def func (self, t, x) :
        pass

    # Abstract method to prepare for the next control interval.
    def prime (self, x) :
        self.control.prime(x)

    def compute_trajectory (self, t0, tf, x0) :
        tu = self.sys.t_spec.tu(t0, tf)

        # xx = np.empty((tu.shape[0]+1,tu.shape[1],) + (len(x0),),dtype=x0.dtype)
        # xx[0,0,:] = x0
        xx = Trajectory(self.sys.t_spec, t0, x0, tf)

        for i, tt in enumerate(tu) :
            # self.prime(xx[i,0,:])
            self.prime(xx(tt[0]))
            for j, t in enumerate(tt) :
                xx.set(t + self.sys.t_spec.t_step, self.func(t, xx(t)))

                # if j == len(tt)-1 :
                #     xx[i+1,0,:] = xtp1
                # else :
                #     xx[i,j+1,:] = xtp1
        
        return xx
    
    def f_replace (self, x) :
        ret = np.empty_like(x)
        for i in range(len(x)) :
            xi = np.copy(x); xi[i].vec[1] = x[i].vec[0]
            _reti = self.sys.f_i[i] (xi, self.control.iuCALC_x[i,:], self.dist.w(0,xi))[0]
            xi[i].vec[1] = x[i].vec[1]; xi[i].vec[0] = x[i].vec[1]
            ret_i = self.sys.f_i[i] (xi, self.control.iuCALCx_[i,:], self.dist.w(0,xi))[0]
            ret[i] = np.intersection(_reti, ret_i)
        return ret


class NNCSystem (ControlledSystem) :
    def __init__(self, sys:NLSystem, nn:NeuralNetwork, incl_method='jacobian', interc_mode=None,
                 dist:Disturbance=NoDisturbance(1), uclip=np.interval(-np.inf,np.inf)) -> None:
        self.nn = nn
        ControlledSystem.__init__(self, sys, NeuralNetworkControl(nn, interc_mode, uclip=uclip),
                                  interc_mode, dist)
        self.incl_method = incl_method
        self.e = None
        self.uj = None
    
    def prime (self, x):
        self.control.prime(x)
        if self.incl_method == 'jacobian' :
            if self.sys.t_spec.type == 'continuous' :
                self.e = np.zeros(self.sys.f_len, dtype=np.interval)
                self.control.step(0, x)
                self.uj = self.control.iuCALC
    
    def func (self, t, x) :
        # Returns x_{t+1} given x_t (euler disc. for continuous time based on t_spec.t_step)
        # Assumes access to pre-computed control in self.control (call control.step before this)

        self.control.step(t, x)

        # Monotone Inclusion
        if x.dtype == np.interval :
            if self.incl_method == 'jacobian' :
                if self.sys.t_spec.type == 'continuous' :
                    _x, x_ = get_lu(x)
                    _u, u_ = get_lu(self.uj)
                    A, B = self.sys.get_AB(x, self.uj, self.dist.w(t,x))
                    _A, A_ = get_lu(A)
                    _B, B_ = get_lu(B)
                    _Bp, _Bn = d_positive(_B)
                    B_p, B_n = d_positive(B_)
                    
                    # Bounding the difference: error dynamics
                    self.control.step(0, x)
                    self.e = self.e + self.sys.t_spec.t_step * self.f_replace(x)
                    _e, e_ = get_lu(self.e)

                    # Centered around _x
                    _K = _Bp@self.control._C + _Bn@self.control.C_
                    K_ = B_p@self.control.C_ + B_n@self.control._C
                    _Kp, _Kn = d_positive(_K)
                    K_p, K_n = d_positive(K_)

                    _L = _A + _K
                    L_ = A_ + K_
                    _Lp, _Ln = d_metzler(_L)
                    L_p, L_n = d_metzler(L_)
                    f = self.sys.f(_x,_u,self.dist.w(t,_x))[0].reshape(-1)
                    print(- _Kp@e_ - _Kn@_e)
                    d_x1 = _Lp@_x + _Ln@x_ - _Kp@e_ - _Kn@_e - _A@_x - _B@_u + _Bp@self.control._d + _Bn@self.control.d_ + f + _e
                    dx_1 = L_p@x_ + L_n@_x - K_p@_e - K_n@e_ - A_@_x - B_@_u + B_p@self.control.d_ + B_n@self.control._d + f + e_

                    # Centered around x_
                    _K = B_p@self.control._C + B_n@self.control.C_
                    K_ = _Bp@self.control.C_ + _Bn@self.control._C
                    _Kp, _Kn = d_positive(_K)
                    K_p, K_n = d_positive(K_)

                    _L = A_ + _K
                    L_ = _A + K_
                    _Lp, _Ln = d_metzler(_L)
                    L_p, L_n = d_metzler(L_)
                    f = self.sys.f(x_,u_,self.dist.w(t,x_))[0].reshape(-1)
                    d_x2 = _Lp@_x + _Ln@x_ - _Kp@e_ - _Kn@_e - A_@x_ - B_@u_ + B_p@self.control._d + B_n@self.control.d_ + f + _e
                    dx_2 = L_p@x_ + L_n@_x - K_p@_e - K_n@e_ - _A@x_ - _B@u_ + _Bp@self.control.d_ + _Bn@self.control._d + f + e_

                    # Take the intersection of the two decompositions
                    return x + self.sys.t_spec.t_step * np.intersection(get_iarray(d_x1, dx_1),get_iarray(d_x2, dx_2))
                    # raise NotImplementedError('jacobian method for continuous mode not implemented')
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
                    d_x2 = _Lp@_x + _Ln@x_ - A_@x_ - B_@u_ + B_p@self.control._d + B_n@self.control.d_ + f
                    dx_2 = L_p@x_ + L_n@_x - _A@x_ - _B@u_ + _Bp@self.control.d_ + _Bn@self.control._d + f
                    return np.intersection(get_iarray(d_x1, dx_1),get_iarray(d_x2, dx_2))

            elif self.incl_method == 'interconnect' :
                if self.sys.t_spec.type == 'continuous' :
                    return x + self.sys.t_spec.t_step*self.f_replace(x)
                else :
                    # print(x, self.control.iuCALC, self.dist.w(t,x))
                    return self.sys.f(x, self.control.iuCALC, self.dist.w(t,x))[0].reshape(-1)
        # No disturbance system
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
    t_spec = ContinuousTimeSpec(0.01,0.1)
    sys = NLSystem([px, py, psi, v], [u1, u2], [w], f_eqn, t_spec)
    net = NeuralNetwork('../examples/vehicle/models/100r100r2')
    # clsys = NNCSystem(sys, net, 'jacobian', uclip=uclip)
    clsys = NNCSystem(sys, net, 'interconnect', uclip=uclip)

    t_span = [0,1]
    # tt = t_spec.tt(0,1)
    # print(t_spec.tu(0,2))

    from interval import from_cent_pert
    cent = np.array([8,8,-2*np.pi/3,2])
    pert = np.array([0.1,0.1,0.01,0.01])
    # pert = np.array([0.001,0.001,0.001,0.001])
    # pert = np.array([0.01,0.01,0.01,0.01])
    x0 = from_cent_pert(cent, pert)

    import time
    repeat_num = 1
    times = []
    for repeat in range(repeat_num) :
        start = time.time()
        xx = clsys.compute_trajectory(t_span[0],t_span[1],x0)
        end = time.time()
        times.append(end - start)
    
    # print(np.mean(times), '\pm', np.std(times))

    print(xx(np.arange(0,1+0.1,0.1)))
    # print(xx(t_spec.tt(0,1)))
