import gurobipy as grb
import numpy as np
# This example formulates and solves the following simple LP model:
# maximize
# x + 2 y + 3 z
# subject to
# x + y <= 1
# y + z <= 1
M = grb.Model('my_model')
x = M.addMVar(3)
A = np.array([[1,1,0],[0,1,1]])
c = np.array([1,2,3])
b = np.array([1,1])
M.setObjective(c@x, grb.GRB.MAXIMIZE)
M.addConstr(A@x <= b)
M.optimize()
print(M.objVal)
print(x.x)