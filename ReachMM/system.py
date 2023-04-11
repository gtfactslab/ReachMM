import sympy as sp
from ReachMM.neural import *

class MixedMonotoneSystem :
    def __init__(self, x_vars, u_vars, w_vars, f_eqn) -> None:
        self.x_vars = x_vars
        self.u_vars = u_vars
        self.w_vars = w_vars
        self.f_eqn  = f_eqn
        self.x = sp.Matrix(x_vars)
        self.u = sp.Matrix(u_vars)
        self.w = sp.Matrix(w_vars)

        tuple = (x_vars, u_vars, w_vars)
        self.f = sp.lambdify(tuple, self.f_eqn, 'numpy')
        self.Df_x = sp.lambdify(tuple, self.f_eqn.jacobian(x_vars), 'numpy')
        self.Df_u = sp.lambdify(tuple, self.f_eqn.jacobian(u_vars), 'numpy')
        self.Df_w = sp.lambdify(tuple, self.f_eqn.jacobian(w_vars), 'numpy')
        print(self.Df_x, self.Df_u, self.Df_w)
    
    def get_AB (self, x, u, w) :
        return self.Df_x(x, u, w), self.Df_u(x, u, w)

    def get_AB_bounds (self, _x, x_, _u, u_, _w, w_) :
        xc = (_x + x_) / 2
        uc = (_u + u_) / 2
        A, B = self.get_AB(xc, uc, [0])
        pass

class NNCS :
    def __init__(self, system:MixedMonotoneSystem, nn:NeuralNetwork) -> None:
        self.system = system
        self.nn = nn
    
    def d(self, _x, x_, _w, w_) :
        A, B = self.system.get_AB(x, w_)

px, py, psi, v, u1, u2, w = sp.symbols('p_x p_y psi v u1 u2 w')
beta = sp.atan2(sp.tan(u2),2)
f_eqn = sp.Matrix([
    v*sp.cos(psi + beta), 
    v*sp.sin(psi + beta), 
    v*sp.sin(beta),
    u1
])

print(f_eqn)

sys = MixedMonotoneSystem([px, py, psi, v], [u1, u2], [w], f_eqn)

x = [0,0,1,0]
u = [1,0.1]

print(sys.get_AB(x, u, [0]))
