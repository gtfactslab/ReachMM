import numpy as np
import interval
from interval import is_iarray
from ReachMM.utils import d_positive

class Control :
    def __init__(self, u_len, mode='hybrid') :
        self.u_len = u_len
        # global, hybrid, or local
        self.mode = mode
        # calculation buffers
        self.uCALC = None
        self.iuCALC = None
        self.iuCALC_x = None
        self.iuCALCx_ = None

    def u (self, t, x) :
        pass
    
    def prime (self, x) :
        pass

    def step (self, t, x) :
        # Inclusion Function
        if x.dtype == np.interval :
            self.iuCALC = self.u (t, x)
            if self.mode == 'global' :
                return self.iuCALC
            elif self.mode == 'hybrid' or self.mode == 'local' :
                d = len(x)
                if self.iuCALC_x is None :
                    self.iuCALC_x = np.empty((d, self.u_len), dtype=np.interval)
                    self.iuCALCx_ = np.empty((d, self.u_len), dtype=np.interval)
                for i in range(d) :
                    xi = np.copy(x); xi[i].vec[1] = x[i].vec[0]
                    self.prime(xi) if self.mode == 'local' else None
                    self.iuCALC_x [i,:] = self.u (t, xi)
                    xi[i].vec[1] = x[i].vec[1]; xi[i].vec[0] = x[i].vec[1]
                    self.prime(xi) if self.mode == 'local' else None
                    self.iuCALCx_ [i,:] = self.u (t, xi)
                return self.iuCALC_x, self.iuCALCx_

        # Regular PW Control
        else :
            self.uCALC = self.u (t, x)
            return self.uCALC
    
    def __call__(self, t, x) : 
        return self.step(t,x)

class LinearControl (Control) :
    def __init__(self, K, mode='hybrid'):
        super().__init__(K.shape[0], mode)
        self.K = K

    def u (self, t, x) :
        return self.K @ x

class Disturbance :
    def __init__(self, w_len) :
        self.w_len = w_len

    def w  (self, t, x) :
        pass
    
class NoDisturbance (Disturbance) :
    def __init__(self, w_len=1):
        super().__init__(w_len)
        self.wZERO = np.zeros((self.w_len))

    def w  (self, t, x) :
        return self.wZERO.astype(x.dtype)

class ConstantDisturbance (Disturbance) :
    def __init__(self, wCONST, iwCONST):
        wCONST = np.asarray(wCONST)
        super().__init__(len(wCONST))
        self.wCONST  = wCONST
        self.iwCONST = iwCONST

    def w (self, t, x) :
        if x.dtype == np.interval : 
            return self.iwCONST
        else :
            return self.wCONST

    # def cut_all (self) :
    #     partitions = []
    #     part_avg = (self._wCONST + self.w_CONST) / 2
    #     w_wh = np.concatenate((self._wCONST, self.w_CONST))

    #     for part_i in range(2**self.w_len) :
    #         part = np.copy(w_wh)
    #         for ind in range (self.w_len) :
    #             part[ind + self.w_len*((part_i >> ind) % 2)] = part_avg[ind]
    #         partitions.append(ConstantDisturbance(part[:self.w_len],part[self.w_len:],self.mode))

    #     return partitions

