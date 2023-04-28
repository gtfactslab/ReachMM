import numpy as np
import interval
from interval import from_cent_pert
import sympy as sp
from ReachMM.system import *
from ReachMM.utils import run_times
import matplotlib.pyplot as plt

x1, x2, u, w = sp.symbols('x1 x2 u w')
f_eqn = [
    x2,
    u*x2**2 - x1
]

# t_spec = ContinuousTimeSpec(0.01, 0.2)
t_spec = DiscretizedTimeSpec(0.2)
# t_spec = DiscreteTimeSpec()
sys = NLSystem([x1, x2], [u], [w], f_eqn, t_spec)
net = NeuralNetwork('models/nn_1_relu')
clsys = NNCSystem(sys, net, 'jacobian')

# x0 = np.array([ np.interval(0.8,0.81), np.interval(0.5,0.51) ])
x0 = np.array([ np.interval(0.8,0.9), np.interval(0.5,0.6) ])
# x0 = np.array([ 0.85, 0.55 ])
# print(clsys.control.u(0,x0))

xx = clsys.compute_trajectory(0, 6, x0)
xx, times = run_times(100, clsys.compute_trajectory, 0, 6, x0)

print(np.mean(times), '\pm', np.std(times))

# print(xx[:,0,:].reshape(-1,2))

# plt.plot(xx[:-1,:,0].reshape(-1), xx[:-1,:,1].reshape(-1))
# plt.show()