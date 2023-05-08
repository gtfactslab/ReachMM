import numpy as np
import interval
from interval import width, sg_box
from ReachMM import TimeSpec
import shapely.geometry as sg
import shapely.ops as so

class Trajectory :
    def __init__(self, t_spec:TimeSpec, t0, x0, t_alloc=None) -> None:
        self.t_spec = t_spec
        self.t0 = t0
        self.tf = t0
        t_alloc = t0 + 10 if t_alloc is None else t_alloc
        tlen = len(self.t_spec.tt(t0,t_alloc))
        self.xx = np.empty((tlen+1,)+x0.shape, x0.dtype)
        # self.xx = np.empty((tu.shape[0]+1,tu.shape[1],) + (len(x0),),dtype=x0.dtype)
        self._n = lambda t : np.round(t/self.t_spec.t_step).astype(int)
        self.set(t0,x0)

    def set (self, t, x) :
        if self._n(t) > self._n(self.tf) :
            self.tf = t
        self.xx[self._n(t),:] = x

    def __call__(self, t) :
        if np.any(self._n(t) > self._n(self.tf)) or np.any(self._n(t) < self._n(self.t0)) :
            raise Exception(f'trajectory not defined at {t} \\notin [{self.t0},{self.tf}]')
        return self.xx[self._n(t),:]

class Partition :
    def __init__(self, t_spec:TimeSpec, t0, x0) :
        self.t_spec = t_spec
        self.t0 = t0
