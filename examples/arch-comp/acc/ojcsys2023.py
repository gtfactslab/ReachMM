import numpy as np
import interval
from interval import from_cent_pert, get_lu, get_cent_pert
import sympy as sp
from ReachMM.time import *
from ReachMM.system import *
from ReachMM.reach import UniformPartitioner, CGPartitioner
from ReachMM.control import ConstantDisturbance
from ReachMM.utils import run_times
import matplotlib.pyplot as plt

# States 
xlead, vlead, glead, xego, vego, gego = sp.symbols('xlead vlead glead xego vego gego')
x_vars = [xlead, vlead, glead, xego, vego, gego]
# Controls
aego = sp.symbols('aego')
# Disturbances
alead = sp.symbols('alead')
# Constants
u = 0.0001

f_eqn = [
    vlead,
    glead,
    -2*glead + 2*alead - u*vlead**2,
    vego,
    gego,
    -2*gego + 2*aego - u*vego**2
]
g_eqn = [
    vset := 30.0,
    Tgap := 1.4,
    vego,
    Drel := xlead - xego,
    vrel := vlead - vego
]
spec = (Drel - (Dsafe := (Ddefault:=10) + Tgap*vego))
print(spec)
spec_lam = sp.lambdify((x_vars,), spec, 'numpy')

t_spec = ContinuousTimeSpec(0.01,0.1)
sys = System(x_vars, [aego], [alead], f_eqn, t_spec)
net = NeuralNetwork('models/controller_3_20_tanh')
clsys = NNCSystem(sys, net, 'interconnect', 
                  dist=ConstantDisturbance([-2],[np.interval(-2,-2)]),
                  g_tuple=(x_vars,), g_eqn=g_eqn)
t_end = 5

x0 = np.array([
    np.interval(90,110),
    np.interval(32,32.2),
    np.interval(0,0),
    np.interval(10,11),
    np.interval(30,30.2),
    np.interval(0,0)
])
xcent, xpert = get_cent_pert(x0)

partitioner = UniformPartitioner(clsys)
popts = UniformPartitioner.Opts(0, 0)

tt = t_spec.tt(0,t_end)

def run () :
    rs = partitioner.compute_reachable_set(0,t_end,x0,popts)
    safe = rs.check_safety(spec_lam, tt)
    return rs, safe
(rs, safe), times = run_times(1, run)

print(f'Safe: {safe} in {np.mean(times)} \\pm {np.std(times)} (s)')
