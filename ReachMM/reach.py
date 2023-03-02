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

def sg_box (x_xh:ArrayLike, xi=0,yi=1):
    n = len(x_xh) // 2
    Xl, Yl, Xu, Yu = \
        x_xh[xi],   x_xh[yi], \
        x_xh[xi+n], x_xh[yi+n]
    return sg.box(Xl,Yl,Xu,Yu)

class Trajectory :
    def __init__(self, x0:ArrayLike, func:Callable,
                control:ControlFunction, t_step:float=None) -> None:
        self.func = func
        self.control = control
        self.sol = None
        self.x0 = x0
        self.t_step = t_step
        self.u_disc = []
        self.n0 = 0

    def get_sol(self, n) :
        # print(n, self.n0)
        # print(len(self.sol), self.depth, self.subpartitions)
        return self.sol[n - self.n0]
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

        x0 = self(t_span[0])

        self.control.step(t_span[0],x0)
        self.u_disc.append(self.control.u_calc)

        if method == 'euler' :
            if self.sol is None:
                self.sol = [self.x0]
            for n in range(int(t_span[0]/self.t_step),int(t_span[1]/self.t_step)):
                self.sol.append(self.get_sol(n) + self.t_step*self.func(n*self.t_step, self.get_sol(n)))

        else :
            ret = solve_ivp(self.func, t_span, x0,
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
                return self.get_sol(int(t/self.t_step))
        
        return self.x0

    def __call__ (self, t) :
        t = np.asarray(t)
        if t.ndim == 0:
            return self._call_single(t)

        return self.sol(t) if self.t_step is None \
                else np.asarray(self.sol)[(t/self.t_step).astype(int)].T

class Partition :
    def __init__(self, x_xh0:ArrayLike, func:Callable, controlif:ControlInclusionFunction,
                 primer:bool, t_step:float=None, primer_depth:int=0, depth:int=0, n0:int=0) -> None:
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
        self.depth = depth
        self.primer_depth = primer_depth
        self.n0 = n0
    
    def get_sol(self, n) :
        # print(n, self.n0)
        # print(len(self.sol), self.depth, self.subpartitions)
        return self.sol[n - self.n0]

    def growth_event (self, t, x_xht) :
        return 1
    
    def t_max (self) :
        if self.sol is None :
            return -1
        elif self.t_step is None :
            return self.sol.t_max
        else :
            return (len(self.sol) + self.n0) * self.t_step
    
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
                    self.sol.append(self.get_sol(n) + self.t_step*self.func(n*self.t_step, self.get_sol(n)))

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
    
    def integrate_eps (self, t_span, method='euler', eps=5, max_primer_depth=1, max_depth=2, check_contr=0.5):
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
                
                n0 = int(t_span[0]/self.t_step)
                nf = int(t_span[1]/self.t_step)
                for n in range(n0,nf):
                    self.sol.append(self.get_sol(n) + self.t_step*self.func(n*self.t_step, self.get_sol(n)))
                    
                    if self.depth < max_depth and n == int(n0 + check_contr*(nf-n0)) :
                        wt0 = width(self.get_sol(n), eps); mwt0 = np.max(wt0)
                        wtm = width(self.sol[-1], eps); mwtm = np.max(wtm)
                        C = mwtm / mwt0
                        mwtf = (C**((nf-n)/(n-n0))) * mwtm
                        if mwtf > 1 :
                            # print(C,mwtf, nf,n)
                            self.sol = self.sol[:(n0+1 - self.n0)]
                            # print(len(self.sol))
                            if self.primer_depth < max_primer_depth:
                                self.cut_all(True, n0)
                            else :
                                self.cut_all(False, n0)
                            
                            self.integrate_eps(t_span,method,eps,max_depth,check_contr)
                            return

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
                subpart.integrate_eps(t_span, method, eps,max_depth,check_contr)
    
    def cut (self, i, primer:bool = False) :
        if self.subpartitions is None :
            Exception('trying to cut something with subpartitions')
        self.subpartitions = []
        avg = (self.x_xh[i] + self.x_xh[i]) / 2
        part1 = np.copy(self.x_xh); part1[i] = avg
        part2 = np.copy(self.x_xh); part2[i + self.n] = avg
        self.subpartitions.append(Partition(part1, self.d, self.controlif, primer, self.t_step))
        self.subpartitions.append(Partition(part2, self.d, self.controlif, primer, self.t_step))

    def cut_all (self, primer:bool = False, n0:int=0) :
        if self.subpartitions is None :
            self.subpartitions = []
            x_xh = self.x_xh0 if self.sol is None else self.sol[-1]
            part_avg = (x_xh[:self.n] + x_xh[self.n:]) / 2

            primer_depth = self.primer_depth + 1 if primer else self.primer_depth

            for part_i in range(2**self.n) :
                part = np.copy(x_xh)
                for ind in range (self.n) :
                    part[ind + self.n*((part_i >> ind) % 2)] = part_avg[ind]
                self.subpartitions.append(Partition(part, self.func, self.controlif, primer, self.t_step, primer_depth, self.depth+1, n0))
        else :
            for part in self.subpartitions :
                part.cut_all(primer,n0)
    
    def sg_boxes (self, t, xi=0, yi=1) :
        if self.sol is not None:
            if self.t_step is None and t <= self.t_max() :
                return [sg_box(self.sol(t))]
            elif self.t_step is not None and t < self.t_max() :
                return [sg_box(self.get_sol(int(t/self.t_step)))]

        if self.subpartitions is not None :
            boxes = []
            for subpart in self.subpartitions :
                boxes.extend(subpart.sg_boxes(t,xi,yi))
            return boxes
        
        return self.x_xh0
    
    def draw_sg_boxes (self, ax, tt, xi=0, yi=1, color='tab:blue') :
        tt = np.asarray(tt)
        
        for t in tt :
            boxes = self.sg_boxes(t, xi, yi)
            shape = so.unary_union(boxes)
            xs, ys = shape.exterior.xy    
            ax.fill(xs, ys, alpha=0.75, fc='none', ec=color)
            xsb, ysb = sg_box(self(t)).exterior.xy
            ax.fill(xsb, ysb, alpha=0.5, fc='none', ec=color, linestyle='--')
    
    def width(self, scale=None):
        return width(self.interpolants[-1], scale)

    def _call_single(self, t):
        if self.sol is not None:
            if self.t_step is None and t <= self.t_max() :
                return self.sol(t)
            elif self.t_step is not None and t <= self.t_max() :
                return self.get_sol(int(t/self.t_step))

        if self.subpartitions is not None :
            x_xht2_parts = np.array([subpart(t) 
                                    for subpart in self.subpartitions])
            # n = x_xht2_parts.shape[1] // 2
            xt2_min  = np.min(x_xht2_parts[:,:self.n], axis=0)
            xht2_max = np.max(x_xht2_parts[:,self.n:], axis=0)
            return np.concatenate((xt2_min,xht2_max))
        
        return self.x_xh0

    def __call__ (self, t) :
        t = np.asarray(t)
        if t.ndim == 0:
            return self._call_single(t)

        if self.subpartitions is None :
            return self.sol(t) if self.t_step is None \
                    else np.asarray(self.sol)[(t/self.t_step).astype(int) - self.n0].T
        else :
            # print(t[t < int(self.t_max()/self.t_step)])
            if self.sol is not None:
                x_xht1 = self.sol(t[t <= self.t_max()]) if self.t_step is None  \
                        else np.asarray(self.sol)[(t[t <= self.t_max()]/self.t_step).astype(int) - self.n0].T
            else :
                x_xht1 = None
            x_xht2_parts = np.array([subpart(t[t > self.t_max()]) 
                                    for subpart in self.subpartitions])
            n = x_xht2_parts.shape[1] // 2
            xt2_min  = np.min(x_xht2_parts[:,:n,:], axis=0)
            xht2_max = np.max(x_xht2_parts[:,n:,:], axis=0)
            x_xht2 = np.concatenate((xt2_min,xht2_max))
            return x_xht2 if x_xht1 is None else np.hstack((x_xht1, x_xht2))
