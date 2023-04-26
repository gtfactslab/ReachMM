import numpy as np
import interval
import sympy as sp
from ReachMM.system import *

x1, x2, u, w = sp.symbols('x1 x2 u w')
f_eqn = [
    x2,
    u*x2**2 - x1
]

t_spec = ContinuousTimeSpec(0.01, 0.1)
sys = NLSystem([x1, x2], [u], [w], f_eqn, t_spec)
net = NeuralNetwork
