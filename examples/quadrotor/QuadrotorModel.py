import numpy as np
from ReachMM import MixedMonotoneModel

'''
x = [
    px, (position x)
    py, (position y)
    pz, (position z)
    vx, (velocity x)
    vy, (velocity y)
    vz  (velocity x) 
]
u = [
    tan(θ), (tan(pitch))
    tan(φ), (tan(roll))
    τ,      (thrust)
]
'''

g = 9.8

class QuadrotorModel (MixedMonotoneModel) :
    def __init__(self, control=None, control_if=None, u_step=0.1):
        super().__init__(control, control_if, u_step)
    
    def control_lim (self, u) :
        pass
    
    def f(self, x, u) :
        # u1, u2, u3 = u
        u1 = np.clip(u[0],-np.pi/9,np.pi/9)
        u2 = np.clip(u[1],-np.pi/9,np.pi/9)
        u3 = np.clip(u[2],0,2*g)
        px, py, pz, vx, vy, vz = x.ravel()
        xdot = np.array([vx, vy, vz, g*u1, -g*u2, u3 - g])
        return xdot
    
    def d(self, x, u, xhat, uhat) :
        # u1, u2, u3 = u
        # u1h, u2h, u3h = uhat
        u1 = np.clip(u[0],-np.pi/9,np.pi/9)
        u2 = np.clip(u[1],-np.pi/9,np.pi/9)
        u3 = np.clip(u[2],0,2*g)
        u1h = np.clip(uhat[0],-np.pi/9,np.pi/9)
        u2h = np.clip(uhat[1],-np.pi/9,np.pi/9)
        u3h = np.clip(uhat[2],0,2*g)
        px, py, pz, vx, vy, vz = x.ravel()
        pxh, pyh, pzh, vxh, vyh, vzh = xhat.ravel()
        xdot = np.array([vx, vy, vz, g*u1, -g*u2h, u3 - g])
        return xdot
    