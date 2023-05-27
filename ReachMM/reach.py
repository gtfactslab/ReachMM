import numpy as np
from numpy.typing import ArrayLike
import interval
from interval import width, get_half_intervals, as_lu, as_iarray, get_lu, get_iarray
from typing import NamedTuple
from ReachMM import TimeSpec
from ReachMM.system import ControlledSystem
from ReachMM.utils import sg_box, sg_boxes, draw_iarrays
import shapely.geometry as sg
import shapely.ops as so
from tqdm import tqdm
import time
class Partition :
    _id = 0

    def __init__(self, t_spec:TimeSpec, t0, x0, 
                 depth=0, primer_depth=0, primer:bool = False) :
        self.t_spec = t_spec
        self.t0 = t0
        self.tf = t0
        if x0.dtype != np.interval :
            raise Exception('need to initialize partition with an interval dtype')
        self.x0 = x0
        self.depth = depth
        self.primer_depth = primer_depth
        self.primer = primer

        self.xx = [x0]
        self._n = lambda t : np.round((t - self.t0)/self.t_spec.t_step).astype(int)
        self.subpartitions = None

        # print(f'new partition {depth}, {primer_depth}, {primer}')

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
        self.tf = t
        self.xx = self.xx[:self._n(t)+1]

    def half_partition_all (self, primer:bool) :
        if self.subpartitions is None :
            primer_depth = self.primer_depth + 1 if primer else self.primer_depth
            intervals = get_half_intervals(self.xx[-1])
            self.subpartitions = [Partition(self.t_spec, self.tf, i, self.depth+1, primer_depth, primer) \
                                  for i in intervals]
        else :
            for part in self.subpartitions :
                part.half_partition_all (primer)

    def get_all (self, t) :
        if self._n(t) <= self._n(self.tf) :
            return [np.asarray(self.xx)[self._n(t)]]
        elif self.subpartitions is not None :
            boxes = []
            for part in self.subpartitions :
                boxes.extend(part.get_all(t))
            return boxes
        else :
            raise Exception(f'Partition not defined at {t} \\notin [{self.t0}, {self.tf}]')

    def draw_rs (self, ax, tt, xi=0, yi=1, color='tab:blue', **kwargs) :
        tt = np.atleast_1d(tt)
        for t in tt :
            draw_iarrays(ax, self.get_all(t), xi, yi, color, **kwargs)

    def __call__(self,t) :
        t = np.asarray(t)
        tmask = self._n(t) <= self._n(self.tf)
        t1 = t[tmask]
        t2 = t[np.logical_not(tmask)]
        xx1 = np.asarray(self.xx)[self._n(t1)]
        if np.any(np.logical_not(tmask)) :
            if self.subpartitions is None :
                raise Exception(f'Partition not defined at {t2} \\notin [{self.t0}, {self.tf}]')
            # Shape is (subpartitions, time, xlen, 2 (lu))
            xx2_parts = as_lu(np.asarray([(subpart(t2)) for subpart in self.subpartitions]))
            # Shape is (time, xlen)
            xx2_l = np.min(xx2_parts[:,:,:,0], axis=0)
            xx2_u = np.max(xx2_parts[:,:,:,1], axis=0)
            xx2 = get_iarray(xx2_l, xx2_u)
            return np.vstack((xx1, xx2))
        elif np.any(tmask) :
            return xx1.reshape(t.shape + (-1,))
        return np.array([])

class Partitioner :
    def __init__(self, clsys:ControlledSystem) -> None:
        self.clsys = clsys

    def compute_reachable_set(self, t0, tf, x0, opts) -> Partition :
        raise NotImplementedError

class UniformPartitioner (Partitioner) :
    class Opts (NamedTuple) :
        depth: int = 0
        primer_depth: int = 0
    
    def __init__(self, clsys:ControlledSystem) -> None:
        super().__init__(clsys)

    def compute_reachable_set(self, t0, tf, x0, opts:Opts=Opts(0,0)) :
        parent = Partition(self.clsys.sys.t_spec, t0, x0, 
                           depth=0, primer_depth=0, primer=True)
        
        for d in range(opts.depth) :
            parent.half_partition_all(d < opts.primer_depth)
        
        for tt in parent.t_spec.tu(t0, tf) :
            self.integrate_partition(parent, tt)

        return parent

    def integrate_partition (self, partition:Partition, tt) :
        x0 = partition(tt[0])
        if partition.primer :
            # before = time.time()
            self.clsys.control.prime(x0)
            # after = time.time(); self.timer += after - before
        if partition.subpartitions is None :
            self.clsys.control.step(tt[0], x0)
            self.clsys.prime(x0)
            for t in tt :
                partition.set(t + self.clsys.sys.t_spec.t_step, self.clsys.func(t, partition(t)))
        else :
            for subpartition in partition.subpartitions :
                self.integrate_partition(subpartition, tt)
    
class CGPartitioner (Partitioner) :
    class Opts (NamedTuple) :
        eps:ArrayLike
        gamma:float = 0.5
        max_depth:int = -1
        max_primer_depth:int = -1
        max_primer_partitions:int = -1
        max_leaf_partitions:int = -1
        max_partitions:int = -1
        enable_bar:bool = False

    def __init__(self, clsys:ControlledSystem) -> None:
        super().__init__(clsys)
        self.opts = None

    def compute_reachable_set(self, t0, tf, x0, opts:Opts) :
        self.opts = opts
        parent = Partition(self.clsys.sys.t_spec, t0, x0, 
                           depth=0, primer_depth=0, primer=True)
        
        # for d in range(opts.max_depth) :
        #     parent.half_partition_all(d < opts.max_primer_depth)
        
        for tt in tqdm(parent.t_spec.tu(t0, tf), disable=not opts.enable_bar) :
            self.integrate_partition(parent, tt)
        return parent

    def integrate_partition (self, partition:Partition, tt) :
        x0 = partition(tt[0])
        if partition.primer :
            self.clsys.control.prime(x0)
        if partition.subpartitions is None :
            self.clsys.control.step(tt[0], x0)
            self.clsys.prime(x0)
            for t in tt :
                partition.set(t + self.clsys.sys.t_spec.t_step, self.clsys.func(t, partition(t)))
                if partition.depth < self.opts.max_depth and \
                    partition._n(t) >= partition._n(t + self.opts.gamma*self.clsys.sys.t_spec.t_step) :

                    # Contraction Guided Adaptive Partitioning algorithm
                    wt0 = width(x0, self.opts.eps); mwt0 = np.max(wt0)
                    wtm = width(partition(t), self.opts.eps); mwtm = np.max(wtm)
                    C = mwtm / mwt0
                    if (C**(1/self.opts.gamma))*mwtm > 1 :
                        partition.revert(tt[0])
                        sub_primer = partition.primer_depth < self.opts.max_primer_depth
                        if sub_primer and partition.primer :
                            partition.primer = False
                        partition.half_partition_all(sub_primer)
                        self.integrate_partition(partition, tt)
                        return

        else :
            for subpartition in partition.subpartitions :
                self.integrate_partition(subpartition, tt)
