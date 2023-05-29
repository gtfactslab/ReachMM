import numpy as np
import interval
from interval import from_cent_pert, get_lu
import sympy as sp
from ReachMM.system import *
# from ReachMM.utils import run_times
import matplotlib.pyplot as plt

x1, x2, u, w = sp.symbols('x1 x2 u w')
f_eqn = [ x2, u*x2**2 - x1 ]

t_spec = ContinuousTimeSpec(0.01, 0.2)
sys = System([x1, x2], [u], [w], f_eqn, t_spec)
net = NeuralNetwork('models/nn_1_relu')
clsys = NNCSystem(sys, net, 'interconnect')

x0 = np.array([ np.interval(0.8,0.81), np.interval(0.5,0.51) ])
xx = clsys.compute_trajectory(0, 1, x0)
print(xx[:,0,:])

# plt.plot(l[:-1,:,0].reshape(-1), u[:-1,:,1].reshape(-1))
# plt.show()
