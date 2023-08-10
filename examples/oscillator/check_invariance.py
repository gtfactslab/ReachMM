import numpy as np
import interval
from interval import from_cent_pert
import sympy as sp

from ReachMM import DiscreteTimeSpec, ContinuousTimeSpec
from ReachMM import System, NeuralNetwork, NNCSystem, AutonomousSystem
from ReachMM import UniformPartitioner, CGPartitioner
from ReachMM.utils import run_times
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import shapely.geometry as sg
import shapely.ops as so
import polytope

x1, x2 = sp.symbols('x1, x2')
f_eqn = sp.Matrix([-2*x1 + x2, -x1 - 2*x2])

t_spec = ContinuousTimeSpec(0.001,0.001)
sys = AutonomousSystem([x1, x2], f_eqn, t_spec)

cent = np.array([1,1])
pert = np.array([0.01,0.01])

traj = sys.compute_trajectory(0,4,cent)
tt = t_spec.tt(0,3)

x0 = from_cent_pert(cent, pert)
print(sys.sys.f(x0,[0],[0])[0])

plt.plot(traj(tt)[:,0], traj(tt)[:,1])
plt.show()
