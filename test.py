import numpy as np
import sympy as sp
import interval
import time
from numba import jit, njit

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


px, py, psi, v, u1, u2, w = sp.symbols('p_x p_y psi v u1 u2 w')
beta = sp.atan(sp.tan(u2)/2)
f_eqn = [
    v*sp.cos(psi + beta), 
    v*sp.sin(psi + beta), 
    v*sp.sin(beta),
    u1
]

sys = NLSystem([px, py, psi, v], [u1, u2], [w], f_eqn)

x0 = np.array([8,8,-2*np.pi/3,2])
pert = np.array([0.1,0.1,0.01,0.01])

z = np.interval(-0.0001,0.0001)

intx = interval.from_cent_pert(x0, pert)
print(intx)

import inspect

sum = 0

for repeat in range(10000) :
    before = time.time()
    # res = _lambdifygenerated(intx, [z,z],[np.interval(0)])
    res = _lambdifygenerated(x0, [0.0,0],[0])
    after = time.time()
    sum += (after - before)

print (sum)


# print(inspect.getsource(sys.f))
# print(inspect.getsource(sys.Df_x))
# print(sys.f(intx, [z,z], [np.interval(0)]))
# print(_lambdifygenerated(intx, [z,z], [np.interval(0)])[0])

