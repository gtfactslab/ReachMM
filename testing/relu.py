"""
|| (relu(x) - relu(-x)) - x ||_\infty

x \in [_x, x_]
"""

import gurobipy as gp
from gurobipy import GRB
import gurobi_ml as gml
import numpy as np
from ReachMM import NeuralNetwork

_x, x_ = -1, 1
eps = 0.1

m = gp.Model()

x = m.addVar(lb=_x, ub=x_, vtype=GRB.CONTINUOUS, name="x")
y1 = m.addVar(name="y1"); a1 = m.addVar(vtype=GRB.BINARY, name="a1")
y2 = m.addVar(name="y2"); a2 = m.addVar(vtype=GRB.BINARY, name="a2")

obj = (y1 - y2 - x)**2
m.setObjective(obj, GRB.MINIMIZE)

# def add_relu (x, y, a) :
#     m.addConstr(y <= x - _x*(1 - a))
#     m.addConstr(y >= x)
#     m.addConstr(y <= x_*a)
#     m.addConstr(y >= 0)

# add_relu((x-eps), y1, a1)
# add_relu((-x+eps), y1, a1)
nx = m.addVar(name="nx")
m.addConstr(nx == -x)

m.addConstr(y1 == gp.max_([x], constant=0))
m.addConstr(y2 == gp.max_([nx], constant=0))

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.VarName, v.X))

print('Obj: %g' % np.sqrt(m.ObjVal))
