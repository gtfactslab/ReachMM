import numpy as np
from ReachMM import MixedMonotoneModel

'''
x1d = x2
x2d = u
'''

class DoubleIntegratorModel(MixedMonotoneModel) :
    def __init__(self, control=None, control_if=None, u_step=0.1):
        super().__init__(control, control_if, u_step)
    
    def f(self, x, u) :
        return np.array([x[1],u[0]])
    
    def d(self, x, u, xh, uh) :
        return np.array([x[1],u[0]])
