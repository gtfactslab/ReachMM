import numpy as np
import sympy as sp
import interval
import time

a = np.array([
    np.interval(-1,1),
    np.interval(-1,1),
])
b = np.array([
    np.interval(-2,2),
    np.interval(1,2),
])

print(np.dot(a,b))
