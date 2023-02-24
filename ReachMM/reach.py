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
from ReachMM import ControlFunction, ControlInclusionFunction
from ReachMM.utils import run_time

def width (x_xh:ArrayLike, scale=None) :
    n = len(x_xh) // 2
    width = x_xh[n:] - x_xh[:n]
    return width if scale is None else width / scale

class Trajectory :
    def __init__(self, x0:ArrayLike, func:Callable,
                control:ControlFunction, t_step:float=None) -> None:
        self.func = func
        self.control = control
        self.sol = None
        self.x0 = x0
        self.t_step = t_step

    def t_max (self) :
        if self.sol is None :
            return 0
        elif self.t_step is None :
            return self.sol.t_max
        else :
            return len(self.sol) * self.t_step

    def integrate (self, t_span, method='RK45') :
        if self.t_step is None and method == 'euler' :
            Exception(f'Calling {method} method without t_step')
        elif self.t_step is not None and method != 'euler' :
            Exception(f'Calling {method} method with t_step')

        x_xh0 = self(t_span[0])

        self.control.step(t_span[0],x_xh0)

        if method == 'euler' :
            if self.sol is None:
                self.sol = [self.x_xh0]
            for n in range(int(t_span[0]/self.t_step),int(t_span[1]/self.t_step)):
                self.sol.append(self.sol[n] + self.t_step*self.func(n*self.t_step, self.sol[n]))

        else :
            ret = solve_ivp(self.func, t_span, x_xh0,
                            method, dense_output=True)
            if self.sol is None :
                self.sol = ret.sol
            else :
                self.sol = OdeSolution(np.append(self.sol.ts, ret.sol.ts[1:]),
                                    (self.sol.interpolants + ret.sol.interpolants))
    
    def _call_single(self, t):
        if self.sol is not None:
            if self.t_step is None and t <= self.t_max() :
                return self.sol(t)
            elif self.t_step is not None and t < len(self.sol) :
                return self.sol[t]

        if self.subpartitions is not None :
            x_xht2_parts = np.array([subpart(t) 
                                    for subpart in self.subpartitions])
            # n = x_xht2_parts.shape[1] // 2
            xt2_min  = np.min(x_xht2_parts[:,:self.n], axis=0)
            xht2_max = np.max(x_xht2_parts[:,self.n:], axis=0)
            return np.concatenate((xt2_min,xht2_max))
        
        return self.x_xh0

    def __call__ (self, t) :
        t = np.asarray(t) if self.t_step is None else np.asarray(t/self.t_step, dtype=int)
        if t.ndim == 0:
            return self._call_single(t)

        if self.subpartitions is None :
            return self.sol(t) if self.t_step is None else self.sol[t]
        else :
            x_xht1 = self.sol(t[t <= self.t_max]) if self.t_step is None else self.sol[t[t <= int(self.t_max/self.t_step)]] 
            x_xht2_parts = np.array([subpart(t[t > self.t_max]) 
                                    for subpart in self.subpartitions])
            n = x_xht2_parts.shape[1] // 2
            xt2_min  = np.min(x_xht2_parts[:,:n,:], axis=0)
            xht2_max = np.max(x_xht2_parts[:,n:,:], axis=0)
            x_xht2 = np.concatenate((xt2_min,xht2_max))
            return np.hstack((x_xht1, x_xht2))

class Partition :
    def __init__(self, x_xh0:ArrayLike, func:Callable, 
                 controlif:ControlInclusionFunction, primer:bool, t_step:float=None) -> None:
        # super().__init__([t0,t0],[x_xh0,x_xh0])
        self.func = func
        self.controlif = controlif
        self.primer = primer
        self.subpartitions = None
        # self.growth_event.terminal = True
        self.sol = None
        self.x_xh0 = x_xh0
        self.n = len(x_xh0) // 2
        self.t_step = t_step
    
    def growth_event (self, t, x_xht) :
        return 1
    
    def t_max (self) :
        if self.sol is None :
            return -1
        elif self.t_step is None :
            return self.sol.t_max
        else :
            return len(self.sol) * self.t_step
    
    def integrate (self, t_span, method='RK45') :
        if self.t_step is None and method == 'euler' :
            Exception(f'Calling {method} method without t_step')
        elif self.t_step is not None and method != 'euler' :
            Exception(f'Calling {method} method with t_step')

        x_xh0 = self(t_span[0])
        if self.primer :
            self.controlif.prime(x_xh0)

        if self.subpartitions is None :
            self.controlif.step(t_span[0],x_xh0)

            if method == 'euler' :
                if self.sol is None:
                    self.sol = [self.x_xh0]
                for n in range(int(t_span[0]/self.t_step),int(t_span[1]/self.t_step)):
                    self.sol.append(self.sol[n] + self.t_step*self.func(n*self.t_step, self.sol[n]))

            else :
                ret = solve_ivp(self.func, t_span, x_xh0,
                                method, dense_output=True)
                if self.sol is None :
                    self.sol = ret.sol
                else :
                    self.sol = OdeSolution(np.append(self.sol.ts, ret.sol.ts[1:]),
                                        (self.sol.interpolants + ret.sol.interpolants))
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
        self.subpartitions.append(Partition(part1, self.d, self.controlif, primer, self.t_step))
        self.subpartitions.append(Partition(part2, self.d, self.controlif, primer, self.t_step))

    def cut_all (self, primer:bool = False) :
        if self.subpartitions is None :
            self.subpartitions = []
            x_xh = self.x_xh0 if self.sol is None else self(self.t_max())
            part_avg = (x_xh[:self.n] + x_xh[self.n:]) / 2

            for part_i in range(2**self.n) :
                part = np.copy(x_xh)
                for ind in range (self.n) :
                    part[ind + self.n*((part_i >> ind) % 2)] = part_avg[ind]
                self.subpartitions.append(Partition(part, self.func, self.controlif, primer, self.t_step))
        else :
            for part in self.subpartitions :
                part.cut_all(primer)
    
    def width(self, scale=None):
        return width(self.interpolants[-1], scale)

    def _call_single(self, t):
        if self.sol is not None:
            if self.t_step is None and t <= self.t_max() :
                return self.sol(t)
            elif self.t_step is not None and t < len(self.sol) :
                return self.sol[t]

        if self.subpartitions is not None :
            x_xht2_parts = np.array([subpart(t) 
                                    for subpart in self.subpartitions])
            # n = x_xht2_parts.shape[1] // 2
            xt2_min  = np.min(x_xht2_parts[:,:self.n], axis=0)
            xht2_max = np.max(x_xht2_parts[:,self.n:], axis=0)
            return np.concatenate((xt2_min,xht2_max))
        
        return self.x_xh0

    def __call__ (self, t) :
        if self.t_step is None:
            t = np.asarray(t)
        else :
            t = np.asarray(t)/self.t_step
            t = t.astype(int)

        if t.ndim == 0:
            return self._call_single(t)

        if self.subpartitions is None :
            return self.sol(t) if self.t_step is None else np.asarray(self.sol)[t].T
        else :
            # print(t[t < int(self.t_max()/self.t_step)])
            if self.sol is not None:
                x_xht1 = self.sol(t[t <= self.t_max()]) if self.t_step is None  \
                        else np.asarray(self.sol)[t[t < int(self.t_max()/self.t_step)]]
            else :
                x_xht1 = None
            x_xht2_parts = np.array([subpart(t[t > self.t_max()]) if self.t_step is None \
                                    else subpart(t[t >= int(self.t_max()/self.t_step)])
                                    for subpart in self.subpartitions])
            n = x_xht2_parts.shape[1] // 2
            xt2_min  = np.min(x_xht2_parts[:,:n,:], axis=0)
            xht2_max = np.max(x_xht2_parts[:,n:,:], axis=0)
            x_xht2 = np.concatenate((xt2_min,xht2_max))
            return x_xht2 if x_xht1 is None else np.hstack((x_xht1, x_xht2))
