import numpy as np
import interval
from interval import width, sg_box
from ReachMM import TimeSpec
# from ReachMM.control import ControlledSystem
import shapely.geometry as sg
import shapely.ops as so

class Trajectory :
    def __init__(self, t_spec:TimeSpec, t0, x0, t_alloc=None) -> None:
        self.t_spec = t_spec
        self.t0 = t0
        self.tf = t0
        t_alloc = t0 + 10 if t_alloc is None else t_alloc
        self.xx = np.empty((self.t_spec.lentt(t0,t_alloc)+1,) + x0.shape, x0.dtype)

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

# class Partition :
#     _id = 0

#     def __init__(self, t_spec:TimeSpec, t0, x0, depth=0, t_alloc=None) :
#         self.t_spec = t_spec
#         self.t0 = t0
#         if x0.dtype != np.interval :
#             raise Exception('need to initialize partition with an interval dtype')
#         self.x0 = x0

#         self.xx = np.empty((self.t_spec.lentt(t0,t_alloc)+1,) + x0.shape, x0.dtype) if t_alloc is not None else None

#         self._n = lambda t : np.round(t/self.t_spec.t_step).astype(int)

#         self.subpartitions = None

#         self._id = Partition._id
#         Partition._id += 1
    
#     # def t_min (self) :


# class Partitioner :
#     def __init__(self, clsys:ControlledSystem) -> None:
#         self.clsys = clsys
    
#     def compute_reachable_set (self, t0, tf) :
#         raise NotImplementedError

# class UniformPartitioner (Partitioner) :
#     def __init__(self, clsys:ControlledSystem) -> None:
#         super().__init__(clsys)

# class CGPartition (Partition) :
#     pass

# class CGPartitioner (Partitioner) :
#     def __init__(self, clsys:ControlledSystem) -> None:
#         super().__init__(clsys)
