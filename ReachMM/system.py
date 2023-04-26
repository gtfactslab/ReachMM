from __future__ import annotations
import numpy as np
import sympy as sp
import interval
from interval import get_lu, get_iarray
from ReachMM.neural import NeuralNetwork, NeuralNetworkControl
from ReachMM.control import Disturbance, NoDisturbance
from ReachMM.utils import d_metzler, d_positive
from pprint import pformat
from numba import jit
from inspect import getsource

class TimeSpec :
    def __init__(self, type, t_step, u_step) -> None:
        # Discrete or Continuous
        self.type = type
        # Euler Integration step
        self.t_step = t_step
        # Control Update time
        self.u_step = u_step

    def tt (self, ti, tf) :
        return np.arange(ti, tf + self.t_step, self.t_step)
    
    def tu (self, ti, tf) :
        return np.arange(ti, tf + self.u_step, self.t_step).reshape((-1,round(self.u_step/self.t_step)))

class DiscreteTimeSpec (TimeSpec) :
    def __init__(self) -> None:
        super().__init__('discrete', 1, 1)

class DiscretizedTimeSpec (TimeSpec) :
    def __init__(self, t_step) -> None:
        super().__init__('discretized', t_step, t_step)

class ContinuousTimeSpec (TimeSpec) :
    def __init__(self, t_step, u_step) -> None:
        if t_step > u_step :
            raise Exception('t_step should be smaller than u_step in ContinuousTimeSpec')
        super().__init__('continuous', t_step, u_step)

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

        print(self.f_eqn.jacobian(x_vars))
        self.Df_x = sp.lambdify(tuple, self.f_eqn.jacobian(x_vars), 'numpy', cse=my_cse)
        # self.Df_x = sp.lambdify(tuple, self.f_eqn.jacobian(x_vars), 'numpy')
        # print(getsource(self.Df_x))
        self.Df_u = sp.lambdify(tuple, self.f_eqn.jacobian(u_vars), 'numpy', cse=my_cse)
        self.Df_w = sp.lambdify(tuple, self.f_eqn.jacobian(w_vars), 'numpy', cse=my_cse)

    def get_AB (self, x, u, w) :
        return self.Df_x(x, u, w)[0], self.Df_u(x, u, w)[0]

class NNCSystem :
    def __init__(self, sys:NLSystem, nn:NeuralNetwork, incl_method='jacobian', cont_mode=None,
                 dist:Disturbance=NoDisturbance(1), uclip=None) -> None:
        self.sys = sys
        self.nn = nn
        if cont_mode is None :
            if self.sys.t_spec.type == 'continuous' :
                cont_mode = 'hybrid'
            else :
                cont_mode = 'global'
        self.control = NeuralNetworkControl(nn, cont_mode, uclip=uclip)
        self.incl_method = incl_method
        self.dist = dist
    
    def func (self, t, x) :
        # Assumes access to pre-computed control in self.control (call control.step before this)

        # Monotone Inclusion
        if x.dtype == np.interval :
            if self.incl_method == 'jacobian' :
                if self.sys.t_spec.type == 'continuous' :
                    raise NotImplementedError('jacobian method for continuous mode not implemented')
                else :
                    _x, x_ = get_lu(x)
                    _u, u_ = get_lu(self.control.iuCALC)
                    # print(x, self.control.iuCALC, self.dist.w(t,x))
                    A, B = self.sys.get_AB(x, self.control.iuCALC, self.dist.w(t,x))
                    # print('A', A)
                    _A, A_ = get_lu(A)
                    _B, B_ = get_lu(B)
                    _Bp, _Bn = d_positive(_B)
                    B_p, B_n = d_positive(B_)
                    _L = _A + _Bp@self.control._C + _Bn@self.control.C_
                    L_ = A_ + B_p@self.control.C_ + B_n@self.control._C
                    _Lp, _Ln = d_positive(_L)
                    L_p, L_n = d_positive(L_)
                    d_x1 = _Lp@_x + _Ln@x_ - _A@_x - _B@_u + self.sys.f(_x,_u,self.dist.w(t,_x))[0].reshape(-1)
                    dx_1 = L_p@x_ + L_n@_x - A_@_x - B_@_u + self.sys.f(_x,_u,self.dist.w(t,_x))[0].reshape(-1)
                    d_x2 = _Lp@_x + _Ln@x_ - _A@x_ - _B@u_ + self.sys.f(x_,u_,self.dist.w(t,x_))[0].reshape(-1)
                    dx_2 = L_p@x_ + L_n@_x - A_@x_ - B_@u_ + self.sys.f(x_,u_,self.dist.w(t,x_))[0].reshape(-1)
                    # return np.intersection(get_iarray(d_x1, dx_1),get_iarray(d_x2, dx_2))
                    return get_iarray(d_x1, dx_1)

                    # print(B.shape, self.control.d.shape)
                    # print('A')
                    # print(A)
                    # print('B')
                    # print(B)
                    # print('C')
                    # print(self.control.C)
                    # print('B@C')
                    # print(B@self.control.C)
                    # # print((A + B@self.control.C))
                    # _ret = (A + B@self.control.C)@x - A@_x - B@_u + B@self.control.d + self.sys.f(_x, _u, self.dist.w(t,_x))[0].reshape(-1)
                    # ret_ = (A + B@self.control.C)@x - A@x_ - B@u_ + B@self.control.d + self.sys.f(x_, u_, self.dist.w(t,x_))[0].reshape(-1)
                    # print(_ret)
                    # print(ret_)
                    # return np.intersection(_ret, ret_)
            elif self.incl_method == 'interconnect' :
                if self.sys.t_spec.type == 'continuous' :
                    ret = np.empty_like(x)
                    for i in range(len(x)) :
                        xi = np.copy(x); xi[i].vec[1] = x[i].vec[0]
                        _reti = self.sys.f_i[i] (xi, self.control.iuCALC_x[i,:], self.dist.w(t,xi))[0]
                        xi[i].vec[1] = x[i].vec[1]; xi[i].vec[0] = x[i].vec[1]
                        ret_i = self.sys.f_i[i] (xi, self.control.iuCALCx_[i,:], self.dist.w(t,xi))[0]
                        # ret[i].vec[0] = _reti.vec[0]
                        # ret[i].vec[1] = ret_i.vec[1]
                        ret[i] = np.intersection(_reti, ret_i)
                    return x + self.sys.t_spec.t_step*ret
                else :
                    # print(x, self.control.iuCALC, self.dist.w(t,x))
                    return self.sys.f(x, self.control.iuCALC, self.dist.w(t,x))[0].reshape(-1)
        else :
            return self.sys.f(x,self.control.uCALC,self.dist(t,x))[0]


if __name__ == '__main__' :
    px, py, psi, v, u1, u2, w = sp.symbols('p_x p_y psi v u1 u2 w')
    beta = sp.atan(sp.tan(u2)/2)
    # f_eqn = [
    #     v*sp.cos(psi + beta), 
    #     v*sp.sin(psi + beta), 
    #     v*sp.sin(beta),
    #     u1
    # ]
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

    t_spec = DiscretizedTimeSpec(0.125)
    # t_spec = ContinuousTimeSpec(0.01,0.25)
    sys = NLSystem([px, py, psi, v], [u1, u2], [w], f_eqn, t_spec)
    net = NeuralNetwork('../examples/vehicle/models/100r100r2')
    clsys = NNCSystem(sys, net, 'jacobian', uclip=uclip)
    # clsys = NNCSystem(sys, net, 'interconnect', uclip=uclip)

    # tt = t_spec.tt(0,1)
    # print(t_spec.tu(0,2))
    tu = t_spec.tu(0,1)

    from interval import from_cent_pert
    cent = np.array([8,8,-2*np.pi/3,2])
    # pert = np.array([0.1,0.1,0.01,0.01])
    pert = np.array([0.001,0.001,0.001,0.001])
    # pert = np.array([0.01,0.01,0.01,0.01])
    x0 = from_cent_pert(cent, pert)

    xx = np.empty((tu.shape[0]+1,tu.shape[1],) + (len(x0),),dtype=np.interval)
    xx[0,0,:] = x0

    import time
    start = time.time()

    for i, tt in enumerate(tu) :
        clsys.control.prime(xx[i,0,:])
        clsys.control.step(0, xx[i,0,:])
        for j, t in enumerate(tt) :
            # x = xx[i,j,:] + t_spec.t_step*clsys.func(t,xx[i,j,:])
            xtp1 = clsys.func(t, xx[i,j,:])
            if j == len(tt)-1 :
                xx[i+1,0,:] = xtp1
            else :
                xx[i,j+1,:] = xtp1

    # print(xx[:,0,:])

    end = time.time()
    print(end-start)

    print(xx[:,0,:])
