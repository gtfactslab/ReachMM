import numpy as np
from numpy.typing import ArrayLike
import interval
from interval import width, get_half_intervals, as_lu, as_iarray, get_lu, get_iarray
from typing import NamedTuple, List
from ReachMM import ContinuousTimeSpec
from ReachMM.system import AutonomousSystem, System, NNCSystem, NeuralNetwork
from ReachMM.utils import sg_box, sg_boxes, draw_iarrays, jordan_canonical, d_metzler
import shapely.geometry as sg
import shapely.ops as so
from tqdm import tqdm
import time
import sympy as sp
import torch
from inclusion import Ordering, standard_ordering
import matplotlib.pyplot as plt

class Paralleletope :
    """Transformed interval representation
    y = Tx
    calX = T^{-1}[y]
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

class WedgeParallaletope :
    class WedgeIdxPair (NamedTuple) :
        x:int
        y:int
    def __init__(self, Tinv:ArrayLike, ) -> None:
        pass

class InvariantSetLocator :
    class Opts (NamedTuple) :
        linearization_pert:ArrayLike = np.interval(-0.1,0.1)
        t_eq:int = 100
        verbose:bool = False
        ordering:Ordering = Ordering(())
        eps:float = 1e-8
        axs:List[plt.Axes] = []

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
        p = self.clsys.sys.ulen
        q = self.clsys.sys.wlen

        dist = self.clsys.dist
        onetraj = self.clsys.compute_trajectory(0,opts.t_eq,np.zeros(n))
        xeq = onetraj(opts.t_eq)
        x0 = xeq + opts.linearization_pert
        ueq = self.clsys.control.u(0,xeq)
        if opts.verbose :
            print(f'xeq: {xeq}'); print(f'x0:  {x0}'); print(f'ueq: {ueq}')
        
        feq = self.clsys.sys.f(xeq, ueq, [0])[0].reshape(-1)
        if opts.verbose :
            print(f'feq: {feq}')

        Aeq, Beq, Deq = self.clsys.sys.get_ABD(xeq, self.clsys.control.u(0,xeq), self.clsys.dist.w(0,xeq))
        if opts.verbose:
            print(f'Aeq: {Aeq}'); print(f'Beq: {Beq}'); print(f'Deq: {Deq}')
        
        self.clsys.control.prime(x0)
        self.clsys.control.step(0,x0)
        Keq = self.clsys.control._C
        Acl = Aeq + Beq@Keq
        if opts.verbose :
            print(f'Keq: {Keq}'); print(f'Acl: {Acl}')
        
        # Acl = U L U^{-1}
        # L = U^{-1} Acl U
 
        # Mixed Jacobian Algorithm
        ordering = standard_ordering(n+p+q)[0] if len(opts.ordering) == 0 else opts.ordering
        _Jx = np.empty((n,n)); J_x = np.empty((n,n))
        _Ju = np.empty((n,p)); J_u = np.empty((n,p))
        _Jw = np.empty((n,q)); J_w = np.empty((n,q))
        xr = np.copy(xeq).astype(np.interval)
        ur = np.copy(ueq).astype(np.interval)
        wr = np.zeros(q).astype(np.interval)

        for j in range(len(ordering)) :
            i = ordering[j]
            if   i < n :
                xr[i] = x0[i]
                _J, J_ = get_lu(self.clsys.sys.Df_x_i[i](xr, ur, wr).astype(np.interval).reshape(-1))
                _Jx[:,i] = _J
                J_x[:,i] = J_
            elif i < n + p :
                k = i - n
                ur[k] = self.clsys.control.uCALC[k]
                _J, J_ = get_lu(self.clsys.sys.Df_u_i[k](xr, ur, wr).astype(np.interval).reshape(-1))
                _Ju[:,k] = _J
                J_u[:,k] = J_
            elif i < n + p + q :
                k = i - n - p
                # wr[k] = w[k]
                _J, J_ = get_lu(self.clsys.sys.Df_w_i[k](xr, ur, wr).astype(np.interval).reshape(-1))
                _Jw[:,k] = _J
                J_w[:,k] = J_

        Jx = get_iarray(_Jx, J_x)
        Ju = get_iarray(_Ju, J_u)
        Jw = get_iarray(_Jw, J_w)
        Arm = Jx + Ju@Keq - Acl
        if opts.verbose :
            print(f'Jx: {Jx}'); print(f'Ju: {Ju}'); print(f'Jw: {Jw}'); print(f'Arm: {Arm}'); 

        e = Arm@x0 - Jx@xeq + Ju@(self.clsys.control.d - ueq) + feq
        if opts.verbose :
            print(f'e: {e}')

        L, U = np.linalg.eig(Acl)
        if opts.verbose :
            print(f'L: {L}'); print(f'U: {U}')
        
        rL = np.real(L)
        iL = np.abs(np.imag(L))

        Tinv = np.empty_like(U, dtype=np.float64)

        polar_tuples = []

        skip = False
        for i, l in enumerate(L) :
            v = U[:,i]
            if not skip :
                if np.iscomplex(l) :
                    polar_tuples.append((i,i+1))
                    rel = np.real(l)
                    iml = np.imag(l)
                    rev = np.real(v)
                    imv = np.imag(v)
                    Tinv[:,i] = -rev
                    Tinv[:,i+1] = imv
                    skip = True
                else :
                    Tinv[:,i] = v
            else :
                skip = False
        
        Tinv[np.abs(Tinv) < opts.eps] = 0
        T = np.linalg.inv(Tinv); T[np.abs(T) < opts.eps] = 0
        At = T@Acl@Tinv; At[np.abs(At) < opts.eps] = 0
        At_sym = sp.Matrix(At)
        Te = T@e
        _Te, Te_ = get_lu(Te)

        if opts.verbose :
            print(f'T: {T}'); print(f'Tinv: {Tinv}')
            print(f'At = T Acl T^{{-1}}: {At}')
            print(f'Te: {Te}')

        t = sp.symbols('t')
        y_vars = sp.Matrix([sym(t) for sym in sp.symbols(f'y[:{n}]', cls=sp.Function)])
        if opts.verbose :
            print(y_vars)

        # for i, yi in enumerate(y_vars) :
        #     print(At_sym[i,:])
        #     print((At_sym[i,:]@y_vars)[0,0])
        #     y_vars[i].fdiff = lambda k : (At_sym[i,:]@y_vars)[0,0]
        #     print(y_vars.diff(t))
        # fdiffs = [lambda i : (At_sym[k,:]@y_vars)[0,0] for k in range(n)]
        # print(fdiffs)
        
        # for k in range(n) :
        #     y_vars[k].fdiff = lambda i : (At_sym[k,:]@y_vars)[0,0]

        y_vars[0].fdiff = lambda i : (At_sym[0,:]@y_vars)[0,0]
        y_vars[1].fdiff = lambda i : (At_sym[1,:]@y_vars)[0,0]
        y_vars[2].fdiff = lambda i : (At_sym[2,:]@y_vars)[0,0]
        y_vars[3].fdiff = lambda i : (At_sym[3,:]@y_vars)[0,0]

        print(y_vars)
        print(y_vars.diff(t))

        # mx1 = rL[0]
        # my1 = iL[0]
        # mx2 = rL[2]
        # my2 = iL[2]
        mx1 = 1
        my1 = 1
        mx2 = 1
        my2 = 1

        r  = [sym(t) for sym in sp.symbols(f'r[:{len(polar_tuples)}]', cls=sp.Function)]
        th = [sym(t) for sym in sp.symbols(f'th[:{len(polar_tuples)}]', cls=sp.Function)]

        subs = [(y_vars[0], mx1*r[0]*sp.cos(th[0])),(y_vars[1], my1*r[0]*sp.sin(th[0])),
                (y_vars[2], mx2*r[1]*sp.cos(th[1])),(y_vars[3], my2*r[1]*sp.sin(th[1]))]

        r[0].fdiff  = lambda i : sp.simplify(((mx1*y_vars[0])**2 + (my1*y_vars[1])**2).diff(t).subs(subs)/(2*r[0]))
        th[0].fdiff = lambda i : sp.simplify((sp.atan2(my1*y_vars[1], mx1*y_vars[0]).diff(t).subs(subs)))
        r[1].fdiff  = lambda i : sp.simplify(((mx2*y_vars[2])**2 + (my2*y_vars[3])**2).diff(t).subs(subs)/(2*r[1]))
        th[1].fdiff = lambda i : sp.simplify((sp.atan2(my2*y_vars[3], mx2*y_vars[2]).diff(t).subs(subs)))

        z_vars = sp.Matrix([r[0], th[0], r[1], th[1]])
        g_eqn = z_vars.diff(t)
        g_eqn = g_eqn.xreplace(dict([(n,0) for n in g_eqn.atoms(sp.Float) if abs(n) < opts.eps]))

        def for_trans (x) :
            return np.array([np.sqrt((mx1*x[0])**2 + (my1*x[1])**2), np.arctan((my1*x[1])/(mx1*x[0])),
                             np.sqrt((mx2*x[2])**2 + (my2*x[3])**2), np.arctan((my2*x[3])/(mx2*x[2]))])
        def inv_trans (x) :
            return np.array([x[0]*np.cos(x[1]), x[0]*np.sin(x[1]), 
                             x[2]*np.cos(x[3]), x[2]*np.sin(x[3])])

        print(g_eqn)
        print(for_trans(e))

        phie = for_trans(e)
        _phie, phie_ = get_lu(phie)

        t_spec = ContinuousTimeSpec(0.1,0.1)
        g_sys = AutonomousSystem(z_vars, g_eqn, t_spec)
        
        z0 = np.array([
            np.interval(0,0.04261/1.00581350080842),
            np.interval(-np.pi,np.pi),
            np.interval(0,0.1062/0.43401876933699),
            np.interval(-np.pi,np.pi),
        ])
        # z0 = np.array([
        #     np.interval(0.005,0.006),
        #     np.interval(-np.pi,np.pi),
        #     np.interval(0.007,0.008),
        #     np.interval(-np.pi,np.pi),
        # ])

        overset = Tinv@inv_trans(z0) + xeq
        print(f'overset: {overset}')
        print(f'x0: {x0}')
        print(np.subseteq(overset, x0))

        _g, g_ = g_sys.f_replace(z0)
        print(_g + _phie)
        print(g_ + phie_)

        # for i, xr_i in enumerate(xr) :
        #     xr_i.fdiff = lambda i : fr_eqn[i]

        # Atm, Atn = d_metzler(At)
        # y = T@xeq + np.array([
        #     np.interval(-0.05,0.05),
        #     np.interval(-0.05,0.05),
        #     np.interval(-0.07,0.07),
        #     np.interval(-0.1,0.1)
        # ])
        # # y = T@xeq + opts.linearization_pert
        # # y = T@x0
        # print(np.all(np.subseteq(Tinv@y, x0)))
        # _y, y_ = get_lu(y)
        # _E = Atm@_y + Atn@y_ + _Te
        # E_ = Atn@_y + Atm@y_ + Te_

        # print(f'_E: {_E}')
        # print(f'E_: {E_}')

        # if np.all(_E >= 0) and np.all(E_ <= 0) :
        #     print('SE >= 0')

        # if len(opts.axs) > 0 :
        #     opts.axs[0].plot()

        # net = NeuralNetwork(self.clsys.nn.dir)
        # net.seq.insert(0, torch.nn.Linear(*Tinv.shape))
        # net[0].weight = torch.nn.Parameter(torch.tensor(Tinv.astype(np.float32)))
        # net[0].bias = torch.nn.Parameter(torch.zeros(n,dtype=torch.float32))

        # y_vars = sp.symbols(f'y[:{n}]')
        # if opts.verbose :
        #     print(y_vars)

        # xr = tuple(Tinv@sp.Matrix(y_vars))
        # g_eqn = sp.Matrix(self.clsys.sys.f_eqn)
        # for i in range(len(xr)) :
        #     g_eqn = g_eqn.subs(self.clsys.sys.x_vars[i], xr[i])
        # g_eqn = T@g_eqn

        # if opts.verbose :
        #     print(f'g_eqn = Tf(T^{{-1}}x): {g_eqn}')

        # sys = System(y_vars, self.clsys.sys.u_vars, self.clsys.sys.w_vars, g_eqn, self.clsys.sys.t_spec)
        # clsys = NNCSystem(sys, net, incl_opts=self.clsys.incl_opts)
        

