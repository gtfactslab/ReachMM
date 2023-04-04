import numpy as np
from numpy import sin, cos, tan, arctan
from ReachMM import MixedMonotoneModel, ControlFunction, ControlInclusionFunction
from ReachMM import DisturbanceFunction, DisturbanceInclusionFunction
from ReachMM.decomp import d_sin, d_cos, d_b1b2
from casadi import *

'''

'''

class LinVehicleModel :
    def __init__(self, control: ControlFunction = None, control_if: ControlInclusionFunction = None, u_step=2.5e-1, 
                 lf=1, lr=1,
                 disturbance: DisturbanceFunction = None, disturbance_if: DisturbanceInclusionFunction = None):
        super().__init__(control, control_if, u_step, disturbance, disturbance_if)
        self.lf = lf
        self.lr = lr
        self.u_step = u_step

    def f (self, x, u, w) :
        u1, u2 = u
        X, Y, phi, v = x.ravel()
        beta = arctan2((self.lf * tan(u2)),(self.lf + self.lr))
        dbeta = ((self.lf/(self.lr + self.lf))*(1/(cos(u2)**2)))/(1 + ((self.lf/(self.lr + self.lf)) * tan(u2))**2)

        A = np.array([
            [0,0, -v*sin(phi + beta), cos(phi + beta)],
            [0,0,  v*cos(phi + beta), sin(phi + beta)],
            [0,0,0,(1/self.lr)*sin(beta)],
            [0,0,0,0]
        ])

        B = np.array([
            [0, -v*sin(phi + beta)*dbeta],
            [0,  v*cos(phi + beta)*dbeta],
            [0, (v/self.lr)*cos(u2)*dbeta],
            [1, 0]
        ])

        return A@x + B@u
