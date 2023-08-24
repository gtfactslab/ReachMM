import numpy as np
import interval
import sympy as sp
import ReachMM as RMM

x0 = np.array([
    np.interval(-1,1),
    np.interval(-1,1)
])
w0 = np.array([ np.interval(-0.1, 0.1) ])
C = np.array([[-2,-3]])

x1, x2 = sp.symbols('x1 x2')
u, w = sp.symbols('u w')
x = sp.Matrix([x1, x2])
f_eqn = sp.Matrix([x2, -x2**3 + u])
# f_eqn = f_eqn.subs(u, (-2*x1 - 3*x2 + w))
print(f_eqn)
t_spec = RMM.ContinuousTimeSpec(0.1,0.1)

sys = RMM.System(x, [u], [w], f_eqn, t_spec)
Acl = sys.Df_x([0,0],[0],[0])[0] + sys.Df_u([0,0],[0],[0])[0]@C
print(f'Acl: {Acl}')
Jx = sys.Df_x(x0,[0],w0)[0]
Ju = sys.Df_u(x0,[0],w0)[0]
print(f'Jx: {Jx}')
print(f'Ju: {Ju}')
print(f'JuC: {Ju@C}')
Arm = (Jx + Ju@C - Acl)
print(f'Arm: {Arm}')
e = Arm@x0 + Ju@(w0)
print(f'e: {e}')

T = np.array([[2,1],[1,1]])
Tinv = np.linalg.inv(T)
y1, y2 = sp.symbols('y1 y2')
y = sp.Matrix([y1, y2])
xr = Tinv@y
y0 = T@x0
print(f'y0: {y0}')

print(sp.Matrix(C))
g_eqn = sp.simplify(T@f_eqn.subs(u,(sp.Matrix(C)@x)[0])).subs(x1, xr[0]).subs(x2, xr[1])
print(g_eqn)
T_sys = RMM.System(y, [u], [w], g_eqn, t_spec)
T_Acl = T_sys.Df_x([0,0],[0],[0])[0] + T_sys.Df_u([0,0],[0],[0])[0]@C@Tinv
print(f'T_Acl: {T_Acl}')
print(f'{T_sys.Df_x_sym}')
T_Jx = T_sys.Df_x(y0,[0],w0)[0]
T_Ju = T_sys.Df_u(y0,[0],w0)[0]
print(f'T_Jx: {T_Jx}')
print(f'T_Ju: {T_Ju}')
print(f'T_JuCTinv: {T_Ju@C@Tinv}')
print(T_Jx + T_Ju@C@Tinv)
# print(f'Tinv d: {Tinv@w0}')
print(T.shape, e.shape)
print(f'Te: {T@e}')

print(f'{T@Jx@Tinv}')
print(f'Tw0: {T@Ju@w0}')
