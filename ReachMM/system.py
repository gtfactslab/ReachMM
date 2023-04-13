import numpy as np
import sympy as sp
from inspect import getsourcefile, getabsfile, getfile
from interval import interval, imath
from ReachMM.neural import *

class IntervalMatrix :
    def __init__(self, intervals) :
        self.intervals = intervals
        self.l = np.array([ [interval(b)[0][0] for b in a] for a in intervals ])
        self.u = np.array([ [interval(b)[0][1] for b in a] for a in intervals ])

class IntervalVector :
    def __init__(self, l, u) :
        self.l = l
        self.u = u
        self.c = (l + u) / 2
        self.intervals = [interval([l[i], u[i]]) for i in range(len(l))]

class MixedMonotoneSystem :
    def __init__(self, x_vars, u_vars, w_vars, f_eqn) -> None:
        self.x_vars = sp.Matrix(x_vars)
        self.u_vars = sp.Matrix(u_vars)
        self.w_vars = sp.Matrix(w_vars)
        self.f_eqn  = sp.Matrix(f_eqn)

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
        # print(self.Df_x, self.Df_u, self.Df_w)
    
    def get_AB (self, x, u, w) :
        return self.Df_x(x, u, w), self.Df_u(x, u, w)

    def get_AB_bounds (self, _x, x_, _u, u_, _w, w_, intervals=False) :
        x = IntervalVector(_x, x_)
        u = IntervalVector(_u, u_)
        w = IntervalVector(_w, w_)
        intA = self.imath_Df_x(x.intervals, u.intervals, w.intervals)[0]
        intB = self.imath_Df_u(x.intervals, u.intervals, w.intervals)[0]
        if intervals :
            return intA, intB
        return intA.l, intA.u, intB.l, intB.u
    
    def d (self, _x, x_, _u, u_, _w, w_) :
        _A, A_, _B, B_ = self.get_AB_bounds(_x, x_, _u, u_, _w, w_)
        return 

class NNCS :
    def __init__(self, system:MixedMonotoneSystem, nn:NeuralNetwork, method='jacobian') -> None:
        self.system = system
        self.nn = nn
        self.control = NeuralNetworkControl(nn)
    
    def d(self, _x, x_, _w, w_) :
        pass


if __name__ == '__main__' :
    px, py, psi, v, u1, u2, w = sp.symbols('p_x p_y psi v u1 u2 w')
    # beta = sp.atan2(sp.tan(u2),2)
    beta = sp.atan(sp.tan(u2)/2)
    f_eqn = [
        v*sp.cos(psi + beta), 
        v*sp.sin(psi + beta), 
        v*sp.sin(beta),
        u1
    ]
    # f_eqn = sp.Matrix([
    #     v*sp.cos(psi + beta), 
    #     v*sp.sin(psi + beta), 
    #     v*sp.sin(beta),
    #     u1
    # ])

    # print(f_eqn)

    sys = MixedMonotoneSystem([px, py, psi, v], [u1, u2], [w], f_eqn)

    x = [0,0,1,0]
    _u = [1,0.1]


    imath_x = [interval([0,1]), interval([1,2]), interval[-0.1,0.1], interval([0,1])]
    imath_u = [interval([0,1]), interval([1,2])]
    imath_w = [0]
    eps = np.array([0.01,0.01,0.001,0.001])
    x = np.array([0.5, 1.5, 0, 0.5]); 
    _u = np.array([0.5, 1.5])
    w = np.array([0])

    _A, A_, _B, B_ = sys.get_AB_bounds(x-eps, x+eps, _u-eps[2:], _u+eps[2:], w, w)
    print(_A)
    print(A_)
    print(_B)
    print(B_)

    # imath_a = sys.imath_Df_x(imath_x, imath_u, imath_w)[0]
    # print(imath_a)
    # print(imath_a.l)
    # print(imath_a.u)

    # a = sys.Df_x(x, u, w)

    # print(a)

    # print(sys.get_AB(x, u, [0]))
