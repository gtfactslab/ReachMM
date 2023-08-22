import numpy as np
from ReachMM import System, NNCSystem
from ReachMM import NeuralNetwork, ContinuousTimeSpec
from ReachMM import InvariantSetLocator
import sympy as sp

A = sp.Matrix([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
B = sp.Matrix([[0,0],[1,0],[0,0],[0,1]])
x = sp.symbols('x[:4]')
u = sp.symbols('u[:2]')
w = sp.symbols('w[:1]')

f_eqn  = sp.Matrix(A)@sp.Matrix(x) + sp.Matrix(B)@sp.Matrix(u)
t_spec = ContinuousTimeSpec(0.1,0.1)
sys = System(x, u, w, f_eqn, t_spec)
net = NeuralNetwork('models/100r100r2_MPC')
clsys = NNCSystem(sys, net)
clsys.set_four_corners()
clsys.set_standard_ordering()

invset = InvariantSetLocator(clsys)
ret = invset.compute_invariant_set(InvariantSetLocator.Opts(verbose=True))
