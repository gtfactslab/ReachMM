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
from ReachMM import DisturbanceFunction, DisturbanceInclusionFunction
from ReachMM import NoDisturbance, NoDisturbanceIF
from ReachMM.utils import run_time
from ReachMM.decomp import d_positive
import sys
from traceback import print_tb
import networkx as nx
from itertools import chain

def width (x_xh:ArrayLike, scale=None) :
    n = len(x_xh) // 2
    width = x_xh[n:] - x_xh[:n]
    return width if scale is None else width / scale

def volume (x_xh:ArrayLike, scale=None) :
    w = width(x_xh, scale)
    ret = 1
    for wi in w :
        ret *= wi
    return ret

def sg_box (x_xh:ArrayLike, xi=0,yi=1):
    n = len(x_xh) // 2
    Xl, Yl, Xu, Yu = \
        x_xh[xi],   x_xh[yi], \
        x_xh[xi+n], x_xh[yi+n]
    # print(Xl,Yl,Xu,Yu)
    return sg.box(Xl,Yl,Xu,Yu)

class Trajectory :
    def __init__(self, x0:ArrayLike, model, 
                 control:ControlFunction, t_step:float=None) -> None:
        self.model = model
        self.control = control
        self.sol = None
        self.x0 = x0
        self.t_step = t_step
        self.u_disc = []
        self.n0 = 0

    def get_sol(self, n) :
        # print(n, self.n0, len(self.sol))
        # print(len(self.sol))
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
            # print(t_span[0]/self.t_step, t_span[1]/self.t_step)
            # print(round(t_span[0]/self.t_step),round(t_span[1]/self.t_step))
            for n in range(round(t_span[0]/self.t_step),round(t_span[1]/self.t_step)):
                # print(n)
                self.sol.append(self.get_sol(n) + self.t_step*self.model.func_(n*self.t_step, self.get_sol(n)))

        else :
            ret = solve_ivp(self.model.func_, t_span, x0,
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
                # print(t,self.t_step)
                return self.get_sol(int(t/self.t_step))
        
        return self.x0

    def __call__ (self, t) :
        t = np.asarray(t)
        if t.ndim == 0:
            return self._call_single(t)

        return self.sol(t) if self.t_step is None \
                else np.asarray(self.sol)[(t/self.t_step).astype(int)].T

class Partition :
    _id = 0

    def __init__(self, x_xh0:ArrayLike, model,
                 control_if:ControlInclusionFunction, primer:bool, 
                 disturbance_if:DisturbanceInclusionFunction,
                 t_step:float=None, primer_depth:int=0, depth:int=0, n0:int=0) -> None:
        # super().__init__([t0,t0],[x_xh0,x_xh0])
        self.model = model
        self.control_if = control_if
        self.primer = primer
        self.disturbance_if = disturbance_if
        self.subpartitions = None
        # self.growth_event.terminal = True
        self.sol = None
        self.x_xh0 = x_xh0
        self.n = len(x_xh0) // 2
        self.t_step = t_step
        self.depth = depth
        self.primer_depth = primer_depth
        self.n0 = n0
        self._id = Partition._id
        Partition._id += 1
    
    def get_sol(self, n) :
        try :
            return self.sol[n - self.n0]
        except Exception as e :
            print(n, self.n0)
            print(len(self.sol), self.depth, self.subpartitions)
            print(self.sol)
            a
            # print_tb(e.__traceback__)
            sys.exit(1)


    def growth_event (self, t, x_xht) :
        return 1
    
    def t_min (self) :
        if self.sol is None :
            return -1
        elif self.t_step is None :
            return self.sol.t_min
        else :
            return self.n0 * self.t_step
    def t_max (self) :
        if self.sol is None :
            return -1
        elif self.t_step is None :
            return self.sol.t_max
        else :
            return (len(self.sol) - 1 + self.n0) * self.t_step
    
    def integrate (self, t_span, method='RK45') :
        if self.t_step is None and method == 'euler' :
            Exception(f'Calling {method} method without t_step')
        elif self.t_step is not None and method != 'euler' :
            Exception(f'Calling {method} method with t_step')

        x_xh0 = self(t_span[0])
        if self.primer :
            self.control_if.prime(x_xh0)

        self.model.disturbance_if = self.disturbance_if

        if self.subpartitions is None :
            self.control_if.step(t_span[0],x_xh0)

            if method == 'euler' :
                if self.sol is None:
                    self.sol = [self.x_xh0]
                for n in range(round(t_span[0]/self.t_step),round(t_span[1]/self.t_step)):
                    # self.sol.append(self.get_sol(n) + self.t_step*self.model.func_(n*self.t_step, self.get_sol(n)))
                    if self.control_if.mode == 'disclti' :
                        self.sol.append(self.model.func_(n*self.t_step, self.get_sol(n)))
                    else :
                        self.sol.append(self.get_sol(n) + self.t_step*self.model.func_(n*self.t_step, self.get_sol(n)))

            else :
                ret = solve_ivp(self.model.func_, t_span, x_xh0,
                                method, dense_output=True)
                if self.sol is None :
                    self.sol = ret.sol
                else :
                    self.sol = OdeSolution(np.append(self.sol.ts, ret.sol.ts[1:]),
                                        (self.sol.interpolants + ret.sol.interpolants))
        else :
            for subpart in self.subpartitions :
                subpart.integrate(t_span, method)
    
    def integrate_eps (self, t_span, method='euler', eps=5, max_primer_depth=1, max_depth=2, check_contr=0.5, cut_dist=False):
        if self.t_step is None and method == 'euler' :
            Exception(f'Calling {method} method without t_step')
        elif self.t_step is not None and method != 'euler' :
            Exception(f'Calling {method} method with t_step')

        x_xh0 = self(t_span[0])
        if self.primer :
            self.control_if.prime(x_xh0)

        self.model.disturbance_if = self.disturbance_if

        if self.subpartitions is None :
            self.control_if.step(t_span[0],x_xh0)

            if method == 'euler' :
                if self.sol is None:
                    self.sol = [self.x_xh0]
                
                n0 = round(t_span[0]/self.t_step)
                nf = round(t_span[1]/self.t_step)
                olen = len(self.sol)
                for n in range(n0,nf):
                    if self.control_if.mode == 'disclti' :
                        self.sol.append(self.model.func_(n*self.t_step, self.get_sol(n)))
                    else :
                        self.sol.append(self.get_sol(n) + self.t_step*self.model.func_(n*self.t_step, self.get_sol(n)))

                    
                    if self.depth < max_depth and n >= n0 + round(check_contr*(nf-n0)) :
                        wt0 = width(self.get_sol(n), eps); mwt0 = np.max(wt0)
                        wtm = width(self.sol[-1], eps); mwtm = np.max(wtm)
                        C = mwtm / mwt0
                        mwtf = (C**((nf-(n+1))/((n+1)-n0))) * mwtm
                        if mwtf > 1 :
                            # print(mwtf, mwtm, mwt0)
                            # print(f'cutting from depth {self.depth}')
                            # print(C,mwtf, nf,n)
                            # self.sol = self.sol[:(n0+1 - self.n0)]
                            self.sol = self.sol[:olen]
                            # print(len(self.sol))
                            sub_primer = self.primer_depth < max_primer_depth
                            if sub_primer and self.primer :
                                self.primer = False
                            self.cut_all((self.primer_depth < max_primer_depth), cut_dist, n0)
                            
                            self.integrate_eps(t_span,method,eps,max_primer_depth,max_depth,check_contr,cut_dist)
                            return

            else :
                ret = solve_ivp(self.model.func_, t_span, x_xh0,
                                method, dense_output=True)
                if self.sol is None :
                    self.sol = ret.sol
                else :
                    self.sol = OdeSolution(np.append(self.sol.ts, ret.sol.ts[1:]),
                                        (self.sol.interpolants + ret.sol.interpolants))
        else :
            for subpart in self.subpartitions :
                subpart.integrate_eps(t_span, method, eps, max_primer_depth, max_depth, check_contr, cut_dist)
    
    def cut (self, i, primer:bool = False) :
        if self.subpartitions is None :
            Exception('trying to cut something with subpartitions')
        self.subpartitions = []
        avg = (self.x_xh[i] + self.x_xh[i]) / 2
        part1 = np.copy(self.x_xh); part1[i] = avg
        part2 = np.copy(self.x_xh); part2[i + self.n] = avg
        self.subpartitions.append(Partition(part1, self.d, self.control_if, primer, self.t_step))
        self.subpartitions.append(Partition(part2, self.d, self.control_if, primer, self.t_step))

    def cut_all (self, primer:bool = False, cut_dist:bool = False, n0:int=0) :
        if self.subpartitions is None :
            self.subpartitions = []
            x_xh = self.x_xh0 if self.sol is None else self.sol[-1]
            part_avg = (x_xh[:self.n] + x_xh[self.n:]) / 2

            primer_depth = self.primer_depth + 1 if primer else self.primer_depth

            if cut_dist :
                dist_parts = self.disturbance_if.cut_all()
            else :
                dist_parts = [self.disturbance_if]

            for part_i in range(2**self.n) :
                part = np.copy(x_xh)
                for ind in range (self.n) :
                    part[ind + self.n*((part_i >> ind) % 2)] = part_avg[ind]
                for dist_part in dist_parts :
                    self.subpartitions.append(Partition(part, self.model, self.control_if, primer, dist_part, self.t_step, primer_depth, self.depth+1, n0))
        else :
            for part in self.subpartitions :
                part.cut_all(primer,cut_dist,n0)
    
    def repartition (self,t) :
        if self.subpartitions is not None:
            # x_xh = self.x_xh0 if self.sol is None else self.sol[-1]
            x_xh = self(t)
            # print('t_max: ', self.t_max())
            part_avg = (x_xh[:self.n] + x_xh[self.n:]) / 2

            for part_i in range(2**self.n) :
                part = np.copy(x_xh)
                for ind in range (self.n) :
                    part[ind + self.n*((part_i >> ind) % 2)] = part_avg[ind]
                # print('subparts.sol')
                # print(self.subpartitions[part_i])
                # print(self.subpartitions[part_i].sol)
                if self.subpartitions[part_i].sol is None :
                    self.subpartitions[part_i].x_xh0 = part
                else :
                    self.subpartitions[part_i].sol[-1] = part

            for part in self.subpartitions :
                # part.cut_all(primer,cut_dist,len(self.sol)-1)
                part.repartition(t)

    
    def sg_boxes (self, t, xi=0, yi=1, T=None) :
        if self.subpartitions is not None and (t >= self.t_max()) :
            boxes = []
            for subpart in self.subpartitions :
                boxes.extend(subpart.sg_boxes(t,xi,yi,T))
            return boxes

        if self.sol is not None:
            if self.t_step is None and t <= self.t_max() :
                bb = self.sol(t)
                if T is not None :
                    bb = d_positive(T) @ bb
                return [sg_box(bb,xi,yi)]
            elif self.t_step is not None and t <= self.t_max() :
                bb = self.get_sol(round(t/self.t_step))
                if T is not None :
                    bb = d_positive(T) @ bb
                return [sg_box(bb,xi,yi)]

        
        return self.x_xh0
    
    def draw_sg_boxes (self, ax, tt, xi=0, yi=1, color='tab:blue', T=None, draw_bb=False) :
        tt = np.asarray(tt)
        
        for t in tt :
            boxes = self.sg_boxes(t, xi, yi, T)
            shape = so.unary_union(boxes)
            xs, ys = shape.exterior.xy    
            ax.fill(xs, ys, alpha=1, fc='none', ec=color)
            if draw_bb:
                bb = self(t)
                if T is not None:
                    bb = d_positive(T) @ bb
                xsb, ysb = sg_box(bb,xi,yi).exterior.xy
                ax.fill(xsb, ysb, alpha=0.5, fc='none', ec=color, linestyle='--')

    
    def width(self, scale=None):
        return width(self.interpolants[-1], scale)

    def get_max_depth (self, t=None) :
        if self.subpartitions is None :
            if t is None or t >= self.t_min() :
                return self.depth 
            else :
                return -1
        else :
            depths = [p.get_max_depth(t) for p in self.subpartitions]
            depths.append(self.depth)
            return max(depths)
    
    def get_max_primer_depth(self) :
        if self.subpartitions is None :
            return self.primer_depth
        else :
            return max([p.get_max_primer_depth() for p in self.subpartitions])
    
    def get_tree (self) :
        if self.subpartitions is None :
            return None
        else :
            edges = [(self._id, s._id) for s in self.subpartitions]
            edges.extend(chain.from_iterable([y for s in self.subpartitions if (y := s.get_tree()) is not None]))
            return edges

    def draw_tree (self, ax, prog="twopi", args="") :
        # print(self.get_tree())
        G = nx.Graph()
        tree = self.get_tree()
        if tree is not None: 
            G.add_edges_from(tree)
            root = min([a for (a,b) in tree])
            pos = nx.nx_agraph.graphviz_layout(G, prog=prog, root=root, args=args)
            nx.draw(G, pos, ax, node_size=20, with_labels=False)

    def _call_single(self, t):
        # if self.subpartitions is not None :
        #     x_xht2_parts = np.array([subpart(t) 
        #                             for subpart in self.subpartitions])
        #     # n = x_xht2_parts.shape[1] // 2
        #     xt2_min  = np.min(x_xht2_parts[:,:self.n], axis=0)
        #     xht2_max = np.max(x_xht2_parts[:,self.n:], axis=0)
        #     return np.concatenate((xt2_min,xht2_max))
        
        # if self.sol is not None:
        #     if self.t_step is None and t <= self.t_max() :
        #         return self.sol(t)
        #     elif self.t_step is not None and t <= self.t_max() :
        #         return self.get_sol(round(t/self.t_step))

        if self.sol is not None and t <= self.t_max() :
            return self.sol(t) if self.t_step is None \
                    else self.get_sol(round(t/self.t_step))
        elif self.subpartitions is not None :
            x_xht_parts = np.array([subpart(t) for subpart in self.subpartitions])
            n = x_xht_parts.shape[1] // 2
            xt2_min  = np.min(x_xht_parts[:,:n], axis=0)
            xht2_max = np.max(x_xht_parts[:,n:], axis=0)
            x_xht = np.concatenate((xt2_min,xht2_max))
            return x_xht

        return self.x_xh0

    def __call__ (self, t) :
        t = np.asarray(t)
        if t.ndim == 0:
            return self._call_single(t)
            # return self([t])[0]

        if self.subpartitions is None :
            return self.sol(t) if self.t_step is None \
                    else np.asarray(self.sol)[np.round(t/self.t_step).astype(int) - self.n0].T
        else :
            # print(t[t < int(self.t_max()/self.t_step)])
            if self.sol is not None:
                x_xht1 = self.sol(t[t <= self.t_max()]) if self.t_step is None  \
                        else np.asarray(self.sol)[np.round(t[t <= self.t_max()]/self.t_step).astype(int) - self.n0].T
            else :
                x_xht1 = None
            x_xht2_parts = np.array([subpart(t[t > self.t_max()]) 
                                    for subpart in self.subpartitions])
            n = x_xht2_parts.shape[1] // 2
            xt2_min  = np.min(x_xht2_parts[:,:n,:], axis=0)
            xht2_max = np.max(x_xht2_parts[:,n:,:], axis=0)
            x_xht2 = np.concatenate((xt2_min,xht2_max))
            return x_xht2 if x_xht1 is None else np.hstack((x_xht1, x_xht2))
