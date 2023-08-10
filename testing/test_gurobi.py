"""
|| (relu(x) - relu(-x)) - x ||_\infty

x \in [_x, x_]
"""

import gurobipy as gp
from gurobipy import GRB
import gurobi_ml as gml
import numpy as np
import interval
from ReachMM import NeuralNetwork, NeuralNetworkControl

# x0 = np.array([
#     np.interval(,1),
#     np.interval(,1),
#     np.interval(,1),
#     np.interval(,1)
# ])
cent = np.array([8,7,-2*np.pi/3,2])
pert = np.array([0.05,0.05,0.01,0.01])
x0 = interval.from_cent_pert(cent, pert)
_x, x_ = interval.get_lu(x0)

net = NeuralNetwork('../examples/vehicle/models/100r100r2')
# print(net)
# net_control = NeuralNetworkControl(net)
# net_control.prime(x0)
# net_control.step(0,x0)
# C = net_control._C
# print(net_control._C, net_control.C_)
# print(net_control._d, net_control.d_)

m = gp.Model()
m.Params.NonConvex = 2

x = m.addMVar((4,), lb=_x, ub=x_, name="x")
u = m.addMVar((2,), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u")
C = m.addMVar((2,4), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="C")
_d = m.addMVar((2,), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="_d")
d_ = m.addMVar((2,), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d_")
pred_constr = gml.add_predictor_constr(m, net.seq, x, u)
pred_constr.print_stats()

m.addConstr(C@x + _d <= u)
m.addConstr(u <= C@x + d_)

obj = gp.quicksum(d_ - _d)

m.setObjective(obj, GRB.MINIMIZE)
m.optimize()

vars = m.getVars()
print(C.X)


# print(vars["C"])
# print(vars.keys())
# print(m.getVarByName("C"))
# print(m.getVarByName("_d"))
# print(m.getVarByName("d_"))
# m.get

# m.setObjective(objvars[0], GRB.MAXIMIZE)
# m.optimize()
# x = m.addVar(lb=_x, ub=x_, vtype=GRB.CONTINUOUS, name="x")
# y1 = m.addVar(name="y1"); a1 = m.addVar(vtype=GRB.BINARY, name="a1")
# y2 = m.addVar(name="y2"); a2 = m.addVar(vtype=GRB.BINARY, name="a2")

# obj = (y1 - y2 - x)**2
# m.setObjective(obj, GRB.MINIMIZE)

# # def add_relu (x, y, a) :
# #     m.addConstr(y <= x - _x*(1 - a))
# #     m.addConstr(y >= x)
# #     m.addConstr(y <= x_*a)
# #     m.addConstr(y >= 0)

# # add_relu((x-eps), y1, a1)
# # add_relu((-x+eps), y1, a1)
# nx = m.addVar(name="nx")
# m.addConstr(nx == -x)

# m.addConstr(y1 == gp.max_([x], constant=0))
# m.addConstr(y2 == gp.max_([nx], constant=0))

# m.optimize()

# for v in m.getVars():
#     print('%s %g' % (v.VarName, v.X))

# print('Obj: %g' % np.sqrt(m.ObjVal))