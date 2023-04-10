import numpy as np
from ReachMM import MixedMonotoneModel
from ReachMM import ControlFunction, ControlInclusionFunction
from ReachMM import DisturbanceFunction, DisturbanceInclusionFunction
from ReachMM.decomp import d_metzler, d_positive
from numba import jit
from numba import int32, float32

'''
x1d = x2 + 0.5u
x2d = u
'''
# x1+ = x1 + x2 + 0.5u
# x2+ = x2 + u

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

class DoubleIntegratorModel(MixedMonotoneModel) :
    def __init__(self, control: ControlFunction = None, control_if: ControlInclusionFunction = None, u_step=1, 
                 disturbance: DisturbanceFunction = None, disturbance_if: DisturbanceInclusionFunction = None):
        super().__init__(control, control_if, u_step, disturbance, disturbance_if)
        self.A = np.array([[1,1],[0,1]])
        self.B = np.array([[0.5],[1]])
        self.Am, self.An = d_metzler(self.A, True)
        self.Bp, self.Bn = d_positive(self.B, True)

        self.c = np.array([0])

        if control_if.mode == 'disclti' :
            control_if.A = self.A
            control_if.B = self.B
            control_if.Bp, control_if.Bn = self.Bp, self.Bn
    
    @staticmethod
    @jit(nopython=True)
    def f(x, u, w) :
        return np.array([x[1] + 0.5*u[0],u[0]])
    
    @staticmethod
    @jit(nopython=True)
    def d(x, xh, u, uh, w, wh) :
        return np.array([x[1] + 0.5*u[0],u[0]])
        # (self.A + self.Bp@self.control_if.)

    @staticmethod
    @jit(nopython=True)
    def d_i (i, x, xh, u, uh, w, wh) :
        return (x[1] + 0.5*u[0]) if i == 0 else (u[0])
