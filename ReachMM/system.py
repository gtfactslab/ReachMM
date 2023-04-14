from __future__ import annotations
import numpy as np
import sympy as sp
from inspect import getsourcefile, getabsfile, getfile
from interval import interval, imath
from ReachMM.control import Disturbance, NoDisturbance
from ReachMM.neural import *
from ReachMM.decomp import d_metzler, d_positive
from pprint import pformat
from numba import jit

class IntervalMatrix :
    def __init__(self, intervals, l=None, u=None) :
        self.intervals = intervals
        if l is None :
            self.l = np.array([ [interval(b)[0][0] for b in a] for a in intervals ])
            self.u = np.array([ [interval(b)[0][1] for b in a] for a in intervals ])
        else :
            self.l = l
            self.u = u
        self.shape = self.l.shape
    
    @classmethod
    def fromlu (cls, l, u) :
        intervals = [[interval([l[i][j], u[i][j]]) for j in range(l.shape[1])] \
                                                   for i in range(u.shape[0])]
        return cls(intervals, l, u)
    
    def __getitem__ (self, i) :
        return self.intervals[i]

    def __matmul__ (self, B) :
        m,p1 = self.shape
        p2,n = B.shape
        if p1 != p2 :
            Exception(f'matrices don\'t match k1:{p1}, k2:{p2}')
        ret = [[interval(0) for j in range(n)] for i in range(m)]
        for i in range(m) :
            for j in range(n) :
                for k in range(p1) :
                    ret[i][j] = ret[i][j] + self[i][k] * B[k][j]
        return IntervalMatrix(ret)
    
    def __add__ (self, B) :
        m1, n1 = self.shape
        m2, n2 = B.shape
        if m1 != m2 or n1 != n2 :
            raise Exception(f'matrices don\'t match {m1,n1,m2,n2}')
        return IntervalMatrix([[self[i][j] + B[i][j] for j in range(n1)] for i in range(m2)])
    
    def __sub__ (self, B) :
        m1, n1 = self.shape
        m2, n2 = B.shape
        if m1 != m2 or n1 != n2 :
            raise Exception(f'matrices don\'t match {m1,n1,m2,n2}')
        return IntervalMatrix([[self[i][j] - B[i][j] for j in range(n1)] for i in range(m2)])
    
    def __str__ (self) :
        return pformat(self.intervals, width=100)

class IntervalVector :
    def __init__(self, intervals, l=None, u=None) :
        self.intervals = intervals
        if l is None :
            self.l = np.array([interval(b)[0] for b in intervals])
            self.u = np.array([interval(b)[1] for b in intervals])
        else :
            self.l = l
            self.u = u
    
    @classmethod
    def fromlu (cls, l, u) :
        intervals = [interval([l[i], u[i]]) for i in range(len(l))]
        return cls(intervals, l, u)

class NLSystem :
    def __init__(self, x_vars, u_vars, w_vars, f_eqn) -> None:
        self.x_vars = sp.Matrix(x_vars)
        self.u_vars = sp.Matrix(u_vars)
        self.w_vars = sp.Matrix(w_vars)
        self.f_eqn  = sp.Matrix(f_eqn)
        print(self.f_eqn)

        def my_cse(exprs, symbols=None, optimizations=None, postprocess=None,
            order='canonical', ignore=(), list=True) :
            return sp.cse(exprs=exprs, symbols=sp.numbered_symbols('_dum'), optimizations='basic', 
                          postprocess=postprocess, order=order, ignore=ignore, list=list)

        tuple = (x_vars, u_vars, w_vars)

        self.f    = sp.lambdify(tuple, self.f_eqn, 'numpy', cse=my_cse)
        self.Df_x = sp.lambdify(tuple, self.f_eqn.jacobian(x_vars), 'numpy', cse=my_cse)
        self.Df_u = sp.lambdify(tuple, self.f_eqn.jacobian(u_vars), 'numpy', cse=my_cse)
        self.Df_w = sp.lambdify(tuple, self.f_eqn.jacobian(w_vars), 'numpy', cse=my_cse)

        imathmodule = [{'ImmutableDenseMatrix': IntervalMatrix},imath]
        self.imath_f    = sp.lambdify(tuple, self.f_eqn, imathmodule, cse=my_cse)
        self.imath_Df_x = sp.lambdify(tuple, self.f_eqn.jacobian(x_vars), imathmodule, cse=my_cse)
        self.imath_Df_u = sp.lambdify(tuple, self.f_eqn.jacobian(u_vars), imathmodule, cse=my_cse)
        self.imath_Df_w = sp.lambdify(tuple, self.f_eqn.jacobian(w_vars), imathmodule, cse=my_cse)
    
    def get_AB (self, x, u, w) :
        return self.Df_x(x, u, w), self.Df_u(x, u, w)

    def get_AB_bounds (self, _x, x_, _u, u_, _w, w_, intervals=True) :
        x = IntervalVector.fromlu(_x, x_)
        u = IntervalVector.fromlu(_u, u_)
        w = IntervalVector.fromlu(_w, w_)
        intA = self.imath_Df_x(x.intervals, u.intervals, w.intervals)[0]
        intB = self.imath_Df_u(x.intervals, u.intervals, w.intervals)[0]
        if intervals :
            return intA, intB
        return intA.l, intA.u, intB.l, intB.u
    
    def d (self, _x, x_, _u, u_, _w, w_) :
        _A, A_, _B, B_ = self.get_AB_bounds(_x, x_, _u, u_, _w, w_)
        return 
    
class NNCNLSystem :
    def __init__(self, sys:NLSystem, nn:NeuralNetwork, method='jacobian',
                 dist:Disturbance=NoDisturbance(1)) -> None:
        self.sys = sys
        self.nn = nn
        self.control = NeuralNetworkControl(nn)
        self.method = method
        self.dist = dist
    
    def func (self, t, x) :
        return self.sys.f(x,self.control.uCALC,self.dist(t,x))

    def dunc (self, t, _xx_) :
        n = len(_xx_) // 2
        _x = _xx_[:n]; x_ = _xx_[n:]
        if self.method == 'jacobian' :
            intx = IntervalMatrix.fromlu(_x.reshape(-1,1), x_.reshape(-1,1))
            # self.control.step_if(0, _x, x_)
            intC = IntervalMatrix.fromlu(self.control._C, self.control.C_)
            intd = IntervalMatrix.fromlu(self.control._d.reshape(-1,1), self.control.d_.reshape(-1,1))

            _u, u_ = self.control.u_lb.reshape(-1), self.control.u_ub.reshape(-1)
            intA, intB = self.sys.get_AB_bounds(_x, x_, 
                _u, u_,\
                self.dist._w(t,_x,x_), self.dist.w_(t,_x,x_))

            # print(intA, intB, intC, intd)

            n = len(_xx_) // 2

            _Bp, _Bn = d_positive(intB.l, True)
            B_p, B_n = d_positive(intB.u, True)
            
            _M = (intA.l + _Bn@intC.u + _Bp@intC.l)
            M_ = (intA.u + B_p@intC.u + B_n@intC.l)
            _Mm, _Mn = d_metzler(_M, True)
            M_m, M_n = d_metzler(M_, True)

            _d = _Mm@_x + _Mn@x_ - intA.l@_x - intB.l@_u + _Bp@intd.l.reshape(-1) + _Bn@intd.u.reshape(-1) + self.sys.f(_x, _u, [0])[0].reshape(-1)
            d_ = M_m@x_ + M_n@_x - intA.u@_x - intB.u@_u + B_p@intd.u.reshape(-1) + B_n@intd.l.reshape(-1) + self.sys.f(_x, _u, [0])[0].reshape(-1)
            _d = _Mm@_x + _Mn@x_ - intA.l@_x - intB.l@_u + _Bp@intd.l.reshape(-1) + _Bn@intd.u.reshape(-1) + self.sys.f(_x, _u, [0])[0].reshape(-1)
            d_ = M_m@x_ + M_n@_x - intA.u@_x - intB.u@_u + B_p@intd.u.reshape(-1) + B_n@intd.l.reshape(-1) + self.sys.f(_x, _u, [0])[0].reshape(-1)
            return np.concatenate((_d.reshape(-1), d_.reshape(-1)))

            # ret = np.empty(2*n)
            # for i in range(2*n) :
            #     intM = intA + intB @ intC
            #     xcent = (_x + x_) / 2; 
            #     if i < n :
            #         xcent[i%n] = _x[i%n] 
            #     else :
            #         xcent[i%n] = x_[i%n]
            #     ucent = self.control.u(0, xcent)

            #     _Mm, _Mn = d_metzler(intM.l, True)
            #     M_m, M_n = d_metzler(intM.u, True)
            #     _t1 = _Mm@_x + _Mn@x_
            #     t1_ = M_m@x_ + M_n@_x
            #     t1 = IntervalMatrix.fromlu(_t1.reshape(-1,1), t1_.reshape(-1,1))
            #     t2 = intA@xcent.reshape(-1,1)
            #     t3 = intB@intd 
            #     t4 = intB@ucent.reshape(-1,1)
            #     t5 = self.sys.f(xcent, ucent, [0])[0].reshape(-1,1)
            #     print('\n')
            #     print('_x', _x)
            #     print('x_', x_)
            #     print('xc', xcent)
            #     print('uc', ucent)

            #     print('t1', t1)
            #     print('t2', t2)
            #     print('t3', t3)
            #     print('t4', t4)
            #     print('t5', t5)

            #     intf = t1 - t2 + t3 - t4 + t5
            #     print('if', intf)
            #     # intf = intM@intx - intA@xcent.reshape(-1,1) + intB@intd \
            #     #        - intB@ucent.reshape(-1,1) + self.sys.f(xcent, ucent, [0])[0].reshape(-1,1)
            #     # print(intf)
            #     # print(intf.l)
            #     # print(intf.u)
            #     # return np.concatenate((intf.l.reshape(-1),intf.u.reshape(-1)))
            #     res = np.concatenate((intf.l.reshape(-1),intf.u.reshape(-1)))
            #     ret[i] = res[i]
            # return ret
        elif self.method == 'interconnect' :
            n = len(_xx_) // 2
            ret = np.empty(2*n)
            _xi = np.copy(_x); x_i = np.copy(x_)
            for i in range(n) :
                x_i[i] = _x[i]
                intxi = IntervalVector.fromlu(_xi, x_i)
                intui = IntervalVector.fromlu(self.control._uCALC_x[i,:], self.control.u_CALC_x[i,:])
                intwi = IntervalVector.fromlu(self.dist._w(0,_xi,x_i), self.dist.w_(0,_xi,x_i))
                ret[i] = self.sys.imath_f(intxi.intervals, intui.intervals, intwi.intervals)[0].l[i]
                x_i[i] = x_[i]; _xi[i] = x_[i]
                intxi = IntervalVector.fromlu(_xi, x_i)
                intui = IntervalVector.fromlu(self.control._uCALCx_[i,:], self.control.u_CALCx_[i,:])
                intwi = IntervalVector.fromlu(self.dist._w(0,_xi,x_i), self.dist.w_(0,_xi,x_i))
                ret[i+n] = self.sys.imath_f(intxi.intervals, intui.intervals, intwi.intervals)[0].u[i]
            return ret
    # def d(self, _x, x_, _w, w_) :
    #     pass


if __name__ == '__main__' :
    px, py, psi, v, u1, u2, w = sp.symbols('p_x p_y psi v u1 u2 w')
    beta = sp.atan(sp.tan(u2)/2)
    f_eqn = [
        v*sp.cos(psi + beta), 
        v*sp.sin(psi + beta), 
        v*sp.sin(beta),
        u1
    ]

    sys = NLSystem([px, py, psi, v], [u1, u2], [w], f_eqn)
    net = NeuralNetwork('../examples/vehicle/models/100r100r2')
    clsys = NNCNLSystem(sys, net, method='jacobian')
    # clsys = NNCNLSystem(sys, net, method='interconnect')

    imath_x = [interval([0,1]), interval([1,2]), interval[-0.1,0.1], interval([0,1])]
    imath_u = [interval([0,1]), interval([1,2])]
    imath_w = [0]
    eps = np.array([0.01,0.01,0.001,0.001])
    x = np.array([0.5, 1.5, 0, 0.5]); 
    u = np.array([0.5, 1.5])
    w = np.array([0])

    print(sys.f(x, u, w))

    t_step = 0.125
    tt = np.arange(0,1.25+t_step,t_step)

    x0 = np.array([8,8,-2*np.pi/3,2])
    pert = np.array([0.1,0.1,0.01,0.01])
    # pert = np.array([0.001,0.001,0.001,0.001])
    # pert = np.array([0.01,0.01,0.01,0.01])

    n = len(x0)
    _xx_ = np.empty((len(tt), 2*n))
    _xx_[0,:] = np.concatenate((x0 - pert,x0 + pert))

    import time

    before = time.time()
    for i, t in enumerate(tt[:-1]) :
        clsys.control.prime(_xx_[i,:n], _xx_[i,n:])
        clsys.control.step_if(0, _xx_[i,:n], _xx_[i,n:])
        _xx_[i+1,:] = _xx_[i,:] + t_step*clsys.dunc(t, _xx_[i,:])
    after = time.time()
    print(after - before)
    print(_xx_)

    # clsys.dunc(0, np.concatenate((x-eps,x+eps)))


    # intA, intB = sys.get_AB_bounds(x-eps, x+eps, u-eps[2:], u+eps[2:], w, w)

    # imath_a = sys.imath_Df_x(imath_x, imath_u, imath_w)[0]
    # print(imath_a)
    # print(imath_a.l)
    # print(imath_a.u)

    # a = sys.Df_x(x, u, w)

    # print(a)

    # print(sys.get_AB(x, u, [0]))
