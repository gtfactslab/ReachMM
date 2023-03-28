import numpy as np
from ReachMM import MixedMonotoneModel
from ReachMM import ControlFunction, ControlInclusionFunction
from ReachMM import DisturbanceFunction, DisturbanceInclusionFunction
from ReachMM.decomp import d_metzler, d_positive

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

g = 9.81

class QuadrotorModel(MixedMonotoneModel) :
    def __init__(self, control: ControlFunction = None, control_if: ControlInclusionFunction = None, u_step=0.1, 
                 disturbance: DisturbanceFunction = None, disturbance_if: DisturbanceInclusionFunction = None):
        super().__init__(control, control_if, u_step, disturbance, disturbance_if)
        self.A = np.array([
            [1,0,0,0.1,0,0],
            [0,1,0,0,0.1,0],
            [0,0,1,0,0,0.1],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]])
        self.B = np.array([
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0.1*g,0,0],
            [0,-0.1*g,0],
            [0,0,0.1*g]
        ])
        self.c = np.array([ 0, 0, 0, 0, 0, -g ])
        self.Am, self.An = d_metzler(self.A, True)
        self.Bp, self.Bn = d_positive(self.B, True)

        if control_if.mode == 'disclti' :
            control_if.A = self.A
            control_if.B = self.B
            control_if.Bp, control_if.Bn = self.Bp, self.Bn
    
    def f(self, x, u, w) :
        u1 = np.clip(u[0],-np.pi/9,np.pi/9)
        u2 = np.clip(u[1],-np.pi/9,np.pi/9)
        u3 = np.clip(u[2],0,2*g)
        px, py, pz, vx, vy, vz = x.ravel()
        xdot = np.array([vx, vy, vz, g*u1, -g*u2, u3 - g])
        return xdot
    
    def d(self, x, xh, u, uh, w, wh) :
        u1 = np.clip(u[0],-np.pi/9,np.pi/9)
        u2 = np.clip(u[1],-np.pi/9,np.pi/9)
        u3 = np.clip(u[2],0,2*g)
        u1h = np.clip(uh[0],-np.pi/9,np.pi/9)
        u2h = np.clip(uh[1],-np.pi/9,np.pi/9)
        u3h = np.clip(uh[2],0,2*g)
        px, py, pz, vx, vy, vz = x.ravel()
        pxh, pyh, pzh, vxh, vyh, vzh = xh.ravel()
        xdot = np.array([vx, vy, vz, g*u1, -g*u2h, u3 - g])
        return xdot

    # def d_i (self, i, x, xh, u, uh, w, wh) :
    #     return d[i]

