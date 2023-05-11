import numpy as np
import interval
from interval import width, get_half_intervals, as_lu, as_iarray
from ReachMM import TimeSpec
from ReachMM.system import ControlledSystem
import shapely.geometry as sg
import shapely.ops as so

class Partition :
    _id = 0

    def __init__(self, t_spec:TimeSpec, t0, x0, 
                 depth=0, pdepth=0, primer:bool = False,
                 t_alloc=None) :
        self.t_spec = t_spec
        self.t0 = t0
        self.tf = t0
        if x0.dtype != np.interval :
            raise Exception('need to initialize partition with an interval dtype')
        self.x0 = x0
        self.depth = depth
        self.pdepth = pdepth
        self.primer = primer
        self.t_alloc = t_alloc

        self.xx = [x0]
        # self.xx = np.empty((self.t_spec.lentt(t0,t_alloc)+1,) + x0.shape, x0.dtype) \
        #                 if t_alloc is not None else None
        self._n = lambda t : np.round((t - self.t0)/self.t_spec.t_step).astype(int)
        self.subpartitions = None

        self._id = Partition._id
        Partition._id += 1
    
    def set (self, t, x) :
        if len(self.xx) == self._n(t) :
            self.xx.append(x)
            self.tf = t
        else :
            raise Exception(f"_n(t): {self._n(t)} is not n+1: {len(self.xx)}")

    def __call__(self,t) :
        t = np.atleast_1d(t)
        tmask = t <= self.tf
        t1 = t[tmask]
        t2 = t[np.logical_not(tmask)]
        xx1 = np.asarray(self.xx)[self._n(t1)]
        if self.subpartitions is not None :
            xx2_parts = as_lu(np.asarray([(subpart(t2)) for subpart in self.subpartitions]))
            xx2_l = np.min(xx2_parts[:,:,:,0], axis=0)

        if self.subpartitions is None :
            not_def = np.logical_or(self._n(t) > self._n(self.tf), self._n(t) < self._n(self.t0))
            if np.any(not_def) :
                raise Exception(f'trajectory not defined at {t[not_def]} \\notin [{self.t0},{self.tf}]')
            return self.xx[self._n(t),:]
    
    # def finish (self) :
    #     self.xx = np.array(self.xx)
    #     if self.subpartitions is not None :
    #         for subpart in self.subpartitions :
    #             subpart.finish()
    

    # def t_min (self) :

    # def cut_all (self, t0, primer:bool) :
    #     if self.subpartitions is None :
    #         half_intervals = get_half_intervals(self.x0)
    #         self.subpartitions = [Partition(self.t_spec, t0, hi, ) for hi in half_intervals]
    #     else :
    #         for part in self.subpartitions :
    #             part.cut_all(t0, primer)

class Partitioner :
    def __init__(self, clsys:ControlledSystem) -> None:
        self.clsys = clsys
        self.parent = None

    # def compute_reachable_set (self, x0, t0, tf) :
    #     raise NotImplementedError

class UniformPartitioner (Partitioner) :
    def __init__(self, clsys:ControlledSystem) -> None:
        super().__init__(clsys)
    
    def compute_reachable_set(self, t0, tf, x0, depth, primer_depth):
        parent = Partition(self.clsys.sys.t_spec, t0, x0, 
                           depth=0, pdepth=0, primer=True, t_alloc=None)
        

class CGPartition (Partition) :
    pass

class CGPartitioner (Partitioner) :
    def __init__(self, clsys:ControlledSystem) -> None:
        super().__init__(clsys)
