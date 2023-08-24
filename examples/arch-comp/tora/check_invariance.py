import numpy as np
from ReachMM import System, NNCSystem
from ReachMM import NeuralNetwork, ContinuousTimeSpec
from ReachMM import InvariantSetLocator
import sympy as sp

# A = sp.Matrix([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
# B = sp.Matrix([[0,0],[1,0],[0,0],[0,1]])
x = sp.symbols('x[:4]')
u = sp.symbols('u[:1]')
w = sp.symbols('w[:1]')
f_eqn = sp.Matrix([
    x[1],
    -x[0] + 0.1*sp.sin(x[2]),
    x[3],
# 11*sp.tanh(u),
    u[0] - 10,
])
t_spec = ContinuousTimeSpec(0.1,0.1)
sys = System(x, u, w, f_eqn, t_spec)
net = NeuralNetwork('models/controllerTora')
clsys = NNCSystem(sys, net)
clsys.set_four_corners()
clsys.set_standard_ordering()

invset = InvariantSetLocator(clsys)
ret = invset.compute_invariant_wedge_paralleletope(InvariantSetLocator.Opts(
    # x_eq=np.zeros(4),
    initial_pert=np.array([
        np.interval(-0.8,0.8),
        np.interval(-0.8,0.8),
        np.interval(-1,1),
        np.interval(-1,1),
        # np.interval(-0.1,0.1),
        # np.interval(-0.1,0.1),
        # np.interval(-0.3,0.3),
        # np.interval(-0.25,0.25),
    ]),
    linearization_pert=np.array([
        np.interval(-0.1,0.1),
        np.interval(-0.1,0.1),
        np.interval(-0.3,0.3),
        np.interval(-0.25,0.25),
        # np.interval(-0.1,0.1),
        # np.interval(-0.1,0.1),
        # np.interval(-0.1,0.1),
        # np.interval(-0.1,0.1),
    ]),
    verbose=True))
