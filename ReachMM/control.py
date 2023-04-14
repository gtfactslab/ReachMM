import numpy as np
from ReachMM.decomp import d_positive

class Control :
    def __init__(self, u_len, mode='hybrid') :
        self.u_len = u_len
        # joint or element
        # global, hybrid, or local
        self.mode = mode
        self.uCALC  = None
        self._uCALC = None
        self.u_CALC = None
        self._uCALC_x = None
        self.u_CALC_x = None
        self._uCALCx_ = None
        self.u_CALCx_ = None

    def u (self, t, x) :
        pass
    
    def prime (self, _x, x_) :
        pass
    
    def _u  (self, t, _x, x_) :
        pass

    def u_ (self, t, _x, x_) :
        pass

    def step (self, t, x) :
        self.uCALC = self.u (t, x)
        return self.uCALC

    # Default interconnection inclusion function implementation
    def step_if (self, t, _x, x_) :
        if self.mode == 'global' :
            self._uCALC = self._u (t, _x)
            self.u_CALC = self.u_ (t, x_)
            return self._uCALC, self.u_CALC
        elif self.mode == 'hybrid' or self.mode == 'local' :
            d = len(x_)
            if self._uCALC_x is None :
                self._uCALC_x = np.empty((d, self.u_len))
                self.u_CALC_x = np.empty((d, self.u_len))
                self._uCALCx_ = np.empty((d, self.u_len))
                self.u_CALCx_ = np.empty((d, self.u_len))
            _xi = np.copy(_x); x_i = np.copy(x_)
            for i in range(d) :
                x_i[i] = _x[i]
                self.prime(_xi, x_i) if self.mode == 'local' else None
                self._uCALC_x [i,:] = self._u (0,_x,x_)
                self.u_CALC_x [i,:] = self.u_ (0,_x,x_)
                x_i[i] = x_[i]; _xi[i] = x_[i]
                self.prime(_xi, x_i) if self.mode == 'local' else None
                self._uCALCx_ [i,:] = self._u (0,_x,x_)
                self.u_CALCx_ [i,:] = self.u_ (0,_x,x_)
                _xi[i] = _x[i]
            return self._uCALC_x, self.u_CALC_x, self._uCALCx_, self.u_CALCx_

    def __call__(self, t, x) : 
        return self.step(t,x)

class LinearControl (Control) :
    def __init__(self, K, mode='hybrid'):
        super().__init__(K.shape[0], mode)
        self.K = K
        self.Kp, self.Kn = d_positive(K, separate=True)

    def u (self, t, x) :
        return self.K @ x

    def _u(self, t, _x, x_) :
        return self.Kp @ _x + self.Kn @ x_
    
    def u_ (self, t, _x, x_) :
        return self.Kp @ x_ + self.Kn @ _x

class Disturbance :
    def __init__(self, w_len) :
        self.w_len = w_len

    def w  (self, t, x) :
        pass
    
    def _w  (self, t, _x, x_) :
        pass

    def w_ (self, t, _x, x_) :
        pass

class NoDisturbance (Disturbance) :
    def __init__(self, w_len=0):
        super().__init__(w_len)
        self.wZERO = np.zeros((self.w_len))

    def w  (self, t, x) :
        return self.wZERO

    def _w  (self, t, _x, x_) :
        return self.wZERO

    def w_ (self, t, _x, x_) :
        return self.wZERO

class ConstantDisturbance (Disturbance) :
    def __init__(self, wCONST, _wCONST, w_CONST):
        wCONST = np.asarray(wCONST)
        super().__init__(len(wCONST))
        self.wCONST  = wCONST
        self._wCONST = _wCONST
        self.w_CONST = w_CONST

    def w (self, t, x) :
        return self.wCONST

    def _w  (self, t, _x, x_) :
        return self._wCONST

    def w_ (self, t, _x, x_) :
        return self.w_CONST

    def cut_all (self) :
        partitions = []
        part_avg = (self._wCONST + self.w_CONST) / 2
        w_wh = np.concatenate((self._wCONST, self.w_CONST))

        for part_i in range(2**self.w_len) :
            part = np.copy(w_wh)
            for ind in range (self.w_len) :
                part[ind + self.w_len*((part_i >> ind) % 2)] = part_avg[ind]
            partitions.append(ConstantDisturbance(part[:self.w_len],part[self.w_len:],self.mode))

        return partitions

