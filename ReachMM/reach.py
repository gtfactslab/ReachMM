from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import shapely.geometry as sg
import shapely.ops as so
from scipy.integrate import OdeSolution, solve_ivp
from numpy.typing import ArrayLike, DTypeLike
from typing import Callable
from ReachMM import ControlInclusionFunction

def width (x_xh:ArrayLike, scale=None) :
    self.n = len(x_xh) // 2
    width = x_xh[self.n:] - x_xh[:self.n]
    return width if scale is None else width / scale

class Partition :
    def __init__(self, x_xh0:ArrayLike, func:Callable, 
                 controlif:ControlInclusionFunction, primer:bool) -> None:
        # super().__init__([t0,t0],[x_xh0,x_xh0])
        self.func = func
        self.controlif = controlif
        self.primer = primer
        self.subpartitions = None
        # self.growth_event.terminal = True
        self.sol = None
        self.x_xh0 = x_xh0
        self.n = len(x_xh0) // 2
    
    def growth_event (self, t, x_xht) :
        return 1
    
    def integrate (self, t_span, method='RK45') :
        x_xh0 = self.x_xh0 if self.sol is None else self(t_span[0])
        if self.primer :
            self.controlif.prime(x_xh0)
            print('primed with ', x_xh0)

        if self.subpartitions is None :
            self.controlif.step(t_span[0],x_xh0)
            if self.sol is None :
                ret = solve_ivp(self.func, t_span, x_xh0,
                                method, dense_output=True)
                self.sol = ret.sol
            else :
                ret = solve_ivp(self.func, t_span, x_xh0,
                                method, dense_output=True)
                self.sol.ts = np.append(self.sol.ts, ret.sol.ts)
                self.sol.t_max = self.sol.ts[-1]
                self.sol.interpolants.append(ret.sol.interpolants)
        else :
            for subpart in self.subpartitions :
                subpart.integrate(t_span, method)
    
    def cut (self, i, primer:bool = False) :
        if self.subpartitions is None :
            Exception('trying to cut something with subpartitions')
        self.subpartitions = []
        avg = (self.x_xh[i] + self.x_xh[i]) / 2
        part1 = np.copy(self.x_xh); part1[i] = avg
        part2 = np.copy(self.x_xh); part2[i + self.n] = avg
        self.subpartitions.append(Partition(part1, self.d, self.controlif, primer))
        self.subpartitions.append(Partition(part2, self.d, self.controlif, primer))

    def cut_all (self, primer:bool = False) :
        parts = []
        part_avg = (self.x_xh[:self.n] + self.x_xh[self.n:]) / 2

        for part_i in range(2**self.n) :
            part = np.copy(self.x_xh)
            for ind in range (self.n) :
                part[ind + self.n*((part_i >> ind) % 2)] = part_avg[ind]
            parts.append(Partition(part, self.func, self.controlif, primer))
        return parts
    
    def width(self, scale=None):
        return width(self.interpolants[-1], scale)

    def _call_single(self, t):
        if self.subpartitions is not None :
            if t <= self.t_max :
                return self.sol(t)
            x_xht2_parts = np.array([subpart(t) 
                                    for subpart in self.subpartitions])
            # n = x_xht2_parts.shape[1] // 2
            xt2_min  = np.min(x_xht2_parts[:,:self.n], axis=0)
            xht2_max = np.max(x_xht2_parts[:,self.n:], axis=0)
            return np.concatenate((xt2_min,xht2_max))

        if self.sol is None :
            return self.x_xh0

        return self.sol(t)

    def __call__ (self, t) :
        if self.subpartitions is None :
            return self.sol(t)
        else :
            t = np.asarray(t)
            if t.ndim == 0:
                return self._call_single(t)
            x_xht1 = self.sol(t[t <= self.t_max])
            x_xht2_parts = np.array([subpart(t[t > self.t_max]) 
                                    for subpart in self.subpartitions])
            n = x_xht2_parts.shape[1] // 2
            xt2_min  = np.min(x_xht2_parts[:,:n,:], axis=0)
            xht2_max = np.max(x_xht2_parts[:,n:,:], axis=0)
            x_xht2 = np.concatenate((xt2_min,xht2_max))
            return np.hstack((x_xht1, x_xht2))
