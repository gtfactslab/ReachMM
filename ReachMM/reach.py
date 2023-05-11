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
                 depth=0, pdepth=0, primer:bool = False) :
        self.t_spec = t_spec
        self.t0 = t0
        self.tf = t0
        if x0.dtype != np.interval :
            raise Exception('need to initialize partition with an interval dtype')
        self.x0 = x0
        self.depth = depth
        self.pdepth = pdepth
        self.primer = primer

        self.xx = [x0]
        self._n = lambda t : np.round((t - self.t0)/self.t_spec.t_step).astype(int)
        self.subpartitions = None

        self._id = Partition._id
        Partition._id += 1
    
    def set (self, t, x) :
        if len(self.xx) == self._n(t) :
            self.xx.append(x)
            self.tf = t
        else :
            raise Exception(f'_n(t): {self._n(t)} is not n+1: {len(self.xx)}')

    def revert (self, t) :
        if t > self.tf :
            raise Exception(f'cannot revert to a future time t={t} > tf={self.tf}')
        self.xx = self.xx[:self._n(t)]

    def half_partition_all (self, primer:bool) :
        if self.subpartitions is None :
            pdepth = self.pdepth + 1 if primer else self.pdepth
            intervals = get_half_intervals(self.xx[-1])
            self.subpartitions = [Partition(self.t_spec, self.tf, i, self.depth+1, pdepth, primer) \
                                  for i in intervals]
        else :
            for part in self.subpartitions :
                part.half_partition_all (primer)

    def __call__(self,t) :
        t = np.atleast_1d(t)
        tmask = t <= self.tf
        t1 = t[tmask]
        t2 = t[np.logical_not(tmask)]
        xx1 = np.asarray(self.xx)[self._n(t1)]
        if self.subpartitions is None :
            if np.any(np.logical_not(tmask)) :
                raise Exception(f'Partition not defined at {t2} \\notin [{self.t0},{self.tf}]')
            return xx1
        else :
            # Shape is (subpartitions, time, xlen, 2 (lu))
            xx2_parts = as_lu(np.asarray([(subpart(t2)) for subpart in self.subpartitions]))
            # Shape is (time, xlen)
            xx2_l = np.min(xx2_parts[:,:,:,0], axis=0)
            xx2_u = np.max(xx2_parts[:,:,:,1], axis=0)
            xx2 = as_iarray(xx2_l, xx2_u)
            return np.vstack((xx1, xx2))

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

    def integrate_partition (self, partition:Partition, tt) :
        if partition.primer :
            self.clsys.control.prime(partition(tt[0]))
        if partition.subpartitions is None :
            self.clsys.control.step(partition(tt[0]))
            self.clsys.prime(partition(tt[0]))
            for t in tt :
                partition.set(t + self.clsys.sys.t_spec.t_step, self.clsys.func(t, partition(t)))

        else :
            for subpartition in partition.subpartitions :
                self.integrate_partition(subpartition)

class UniformPartitioner (Partitioner) :
    def __init__(self, clsys:ControlledSystem) -> None:
        super().__init__(clsys)
    
    # def integ_partition

    def compute_reachable_set(self, t0, tf, x0, depth, primer_depth):
        parent = Partition(self.clsys.sys.t_spec, t0, x0, 
                           depth=0, pdepth=0, primer=True)
        
        for d in range(depth) :
            parent.half_partition_all(d < primer_depth)

        for tt in parent.t_spec.tu(t0, tf) :
            self.integrate_partition(parent, tt)

class CGPartition (Partition) :
    pass

class CGPartitioner (Partitioner) :
    def __init__(self, clsys:ControlledSystem) -> None:
        super().__init__(clsys)
