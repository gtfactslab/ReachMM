from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import shapely.geometry as sg
import shapely.ops as so
import numpy.typing as npt
from scipy.integrate import solve_ivp
import time
from ReachMM import MixedMonotoneModel

# class Box :
#     def __init__(self, x_xh:npt.ArrayLike) -> None:
#         self.x_xh = x_xh
#         self.n = len(x_xh) // 2
#     def width (self) -> npt.NDArray :
#         x_xh = self.x_xh()
#         n = len(x_xh) // 2
#         return x_xh[n:] - x_xh[:n]
#     def max_width (self) -> npt.DTypeLike :
#         return np.max(self.width())
#     def argmax_width (self) -> int :
#         return np.argmax(self.width())
#     def __len__(self) :
#         return 
#     def __array__(self) :
#         return self.x_xh

class Partition :
    def __init__(self, box0: Box, model: MixedMonotoneModel, t_step:float) -> None:
        boxt = [box0]
        tt = [0]
        self.model = model
    def integrate (self, dt, method='RK45', stopping=None):
        t_span = [self.tt[-1],self.tt[-1]+dt]
        sol = self.model.integrate(box0,)

class ControlPartition :
    def __init__(self, x_xh0:npt.NDArray) -> None:
        
        pass