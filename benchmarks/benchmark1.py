import numpy as np
import interval
from interval import from_cent_pert
import sympy as sp
from ReachMM.system import *
import matplotlib.pyplot as plt

x1, x2, u, w = sp.symbols('x1 x2 u w')
f_eqn = [
    x2,
    (u-4)*x2**2 - x1
]

t_spec = ContinuousTimeSpec(0.01, 0.2)
# t_spec = DiscretizedTimeSpec(0.1)
# t_spec = DiscreteTimeSpec()
sys = NLSystem([x1, x2], [u], [w], f_eqn, t_spec)
net = NeuralNetwork('models/nn_1_relu')
clsys = NNCSystem(sys, net, 'interconnect')
print(clsys)

# x0 = np.array([ np.interval(0.8,0.81), np.interval(0.5,0.51) ])
# x0 = np.array([ np.interval(0.8,0.825), np.interval(0.5,0.525) ])
x0 = np.array([ 0.85, 0.55 ])
print(clsys.control.u(0,x0))

tu = t_spec.tu(0,6)
xx = np.empty((tu.shape[0]+1,tu.shape[1],) + (len(x0),),dtype=x0.dtype)
xx[0,0,:] = x0

import time
repeat_num = 10
times = []
for repeat in range(repeat_num) :
    start = time.time()
    for i, tt in enumerate(tu) :
        clsys.control.prime(xx[i,0,:]) if x0.dtype == np.interval else None
        for j, t in enumerate(tt) :
            clsys.control.step(0, xx[i,j,:])
            # x = xx[i,j,:] + t_spec.t_step*clsys.func(t,xx[i,j,:])
            xtp1 = clsys.func(t, xx[i,j,:])
            if j == len(tt)-1 :
                xx[i+1,0,:] = xtp1
            else :
                xx[i,j+1,:] = xtp1
    end = time.time()
    times.append(end - start)

print(np.mean(times), '\pm', np.std(times))

# print(xx[:,:,:].reshape(-1,2))

plt.plot(xx[:-1,:,0].reshape(-1), xx[:-1,:,1].reshape(-1))
plt.show()
