import numpy as np
from numpy.typing import ArrayLike
import interval
from interval import width, get_half_intervals, as_lu, as_iarray, get_lu, get_iarray
from typing import NamedTuple
from ReachMM import TimeSpec
from ReachMM.system import NNCSystem
from ReachMM.utils import sg_box, sg_boxes, draw_iarrays, jordan_canonical
import shapely.geometry as sg
import shapely.ops as so
from tqdm import tqdm
import time
import sympy as sp

class Paralleletope :
    """Transformed interval representation
    y = Tx
    \\calX = T^{-1}[y]
    """
    def __init__(self, Tinv:ArrayLike, y:ArrayLike) -> None:
        """Constructor for a paralleletope, linear transformation of a hyper-rectangle [y]
        y = Tx
        \calX = T^{-1}[y]

        Args:
            Tinv (ArrayLike): Inverse linear transformation
            y (ArrayLike, dtype=np.interval): Hyper-rectangle in transformed coordinates
        """
        self.Tinv = np.asarray(Tinv)
        self.T = np.linalg.inv(Tinv)
        self.y = np.asarray(y).reshape(-1)
        if self.y.dtype != np.interval :
            raise Exception(f'Trying to construct a Paralleletope with non-interval y: {y}')
        if len(self.y) != self.Tinv.shape[1] :
            raise Exception(f'Trying to construct a Paralleletope with len(y): {len(self.y)} != Tinv.shape[1]: {self.Tinv.shape[1]}')

    def __contains__ (self, x:ArrayLike) :
        """Check to see if T^{-1}[y] contains x.

        Args:
            x (ArrayLike): The vector to check
        """
        return np.all(np.subseteq(self.T@np.asarray(x, dtype=np.interval), self.y))

class InvariantSetLocator :
    class Opts (NamedTuple) :
        linearization_pert:ArrayLike = np.interval(-0.05,0.05)
        t_eq:int = 100
        verbose:bool = False

    def __init__(self, clsys:NNCSystem) -> None:
        self.clsys = clsys
    
    def compute_invariant_set (self, opts:Opts) -> Paralleletope :
        """Compute invariant set of a closed-loop system.

        Args:
            opts (Opts): _description_

        Returns:
            Paralleletope: _description_
        """

        n = self.clsys.sys.xlen
        onetraj = self.clsys.compute_trajectory(0,opts.t_eq,np.zeros(n))
        xeq = onetraj(opts.t_eq)
        x0 = xeq + opts.linearization_pert
        if opts.verbose :
            print(f'xeq: {xeq}'); print(f'x0:  {x0}\n')
        
        Aeq, Beq, Deq = self.clsys.sys.get_ABD(xeq, self.clsys.control.u(0,xeq), self.clsys.dist.w(0,xeq))
        if opts.verbose:
            print(f'Aeq: {Aeq}'); print(f'Beq: {Beq}'); print(f'Deq: {Deq}')
        
        self.clsys.control.prime(x0)
        self.clsys.control.step(0,x0)
        Keq = self.clsys.control._C
        Acl = Aeq + Beq@Keq
        if opts.verbose :
            print(f'Keq: {Keq}'); print(f'Acl: {Acl}')

        P, J = jordan_canonical(Acl, opts.verbose)
        if opts.verbose :
            print(f'P: {P}'); print(f'J: {J}')

        print(P@J@np.linalg.inv(P))


        T = np.linalg.inv(P)
        Tinv = P
        if opts.verbose :
            print(f'T: {T}'); print(f'Tinv: {Tinv}')

        # print(T@Acl@Tinv)

        y_vars = sp.symbols(f'y[:{n}]')
        if opts.verbose :
            print(y_vars)

        xr = tuple(Tinv@sp.Matrix(y_vars))
        g_eqn = sp.Matrix(self.clsys.sys.f_eqn)
