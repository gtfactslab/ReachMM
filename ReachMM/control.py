import numpy as np
from ReachMM.decomp import d_positive

# ControlFunction implements a piecewise constant controller.
class ControlFunction :
    def __init__(self, u_len) :
        self.u_len = u_len
        self.u_calc = None
    
    def u  (self, t, x) :
        pass

    def step(self, t, x) :
        self.u_calc = self.u (t,x)
        return self.u_calc
    
    def __call__(self, t, x) : 
        return self.step(t,x)

class ControlInclusionFunction :
    def __init__(self, u_len, mode='hybrid') :
        self.u_len = u_len
        # joint or element
        # global, hybrid, or local
        self.mode = mode
        self.u_calc  = None
        self.uh_calc = None
        self.u_calc_x   = None
        self.uh_calc_x  = None
        self.u_calc_xh  = None
        self.uh_calc_xh = None
    
    def prime (self, x_xh) :
        pass

    def u  (self, t, x_xh) :
        pass

    def u_i (self, i, t, x_xh, swap_x) :
        pass

    def uh (self, t, x_xh) :
        pass

    def uh_i (self, i, t, x_xh, swap_x) :
        pass

    def step(self, t, x) :
        if self.mode == 'global' :
            self.u_calc  = self.u (t,x)
            self.uh_calc = self.uh(t,x)
            return self.u_calc, self.uh_calc
        elif self.mode == 'hybrid' or self.mode == 'local' :
            d = len(x)//2
            self.u_calc_x   = np.empty((d, self.u_len))
            self.uh_calc_x  = np.empty((d, self.u_len))
            self.u_calc_xh  = np.empty((d, self.u_len))
            self.uh_calc_xh = np.empty((d, self.u_len))
            for i in range(d) :
                self.u_calc_x  [i,:] = self.u_i (i,0,x,False)
                self.uh_calc_x [i,:] = self.uh_i(i,0,x,False)
                self.u_calc_xh [i,:] = self.u_i (i,0,x,True)
                self.uh_calc_xh[i,:] = self.uh_i(i,0,x,True)
            return self.u_calc_x, self.uh_calc_x, self.u_calc_xh, self.uh_calc_xh

    def __call__(self, t, x) : 
        return self.step(t,x)

class LinearControl (ControlFunction) :
    def __init__(self, K):
        super().__init__(K.shape[0])
        self.K = K
    def u (self, t, x) :
        return self.K @ x

class LinearControlIF (ControlInclusionFunction) :
    def __init__(self, K, mode='hybrid'):
        super().__init__(K.shape[0], mode)
        self.K = K
        self.Kp, self.Kn = d_positive(K, separate=True)

    def u(self, t, x_xh) :
        n = len(x_xh) // 2
        x = x_xh[:n]; xh = x_xh[n:]
        return self.Kp @ x + self.Kn @ xh
    
    def uh (self, t, x_xh) :
        n = len(x_xh) // 2
        x = x_xh[:n]; xh = x_xh[n:]
        return self.Kp @ xh + self.Kn @ x
    
    def u_i (self, i, t, x_xh, swap_x) :
        h = len(x_xh) // 2
        x = np.copy(x_xh[:h]); xh = np.copy(x_xh[h:])
        if swap_x :
            x[i] = xh[i]
        else :
            xh[i] = x[i]
        return self.Kp @ x + self.Kn @ xh

    def uh_i (self, i, t, x_xh, swap_x) :
        h = len(x_xh) // 2
        x = np.copy(x_xh[:h]); xh = np.copy(x_xh[h:])
        if swap_x :
            x[i] = xh[i]
        else :
            xh[i] = x[i]
        return self.Kp @ xh + self.Kn @ x

class DisturbanceFunction :
    def __init__(self, w_len) :
        self.w__len = w_len
    
    def w  (self, t, x) :
        pass

    def __call__(self, t, x) : 
        return self.w_(t,x)

class DisturbanceInclusionFunction :
    def __init__(self, w_len, mode='hybrid') :
        self.w__len = w_len
        # joint or element
        # global, hybrid, or local
        self.mode = mode
    
    def w  (self, t, x_xh) :
        pass

    def w_i (self, i, t, x_xh, swap_x) :
        pass

    def wh (self, t, x_xh) :
        pass

    def wh_i (self, i, t, x_xh, swap_x) :
        pass

class NoDisturbance (DisturbanceFunction) :
    def __init__(self, w_len=0):
        super().__init__(w_len)

    def w  (self, t, x_xh) :
        return np.zeros((self.w__len))

    def wh (self, t, x_xh) :
        return np.zeros((self.w__len))

class NoDisturbanceIF (DisturbanceInclusionFunction) :
    def __init__(self, w_len=0):
        super().__init__(w_len)

    def w  (self, t, x_xh) :
        return np.zeros((self.w__len))

    def w_i (self, i, t, x_xh, swap_x) :
        return np.zeros((self.w__len))

    def wh (self, t, x_xh) :
        return np.zeros((self.w__len))

    def wh_i (self, i, t, x_xh, swap_x) :
        return np.zeros((self.w__len))

class ConstantDisturbance (DisturbanceFunction) :
    def __init__(self, w):
        w = np.asarray(w)
        super().__init__(len(w))
        self.w_  = w

    def w (self, t, x) :
        return self.w_

class ConstantDisturbanceIF (DisturbanceInclusionFunction) :
    def __init__(self, w, wh, mode='hybrid'):
        w = np.asarray(w)
        wh = np.asarray(wh)
        super().__init__(len(w), mode)
        self.w_  = w
        self.wh_ = wh

    def w (self, t, x_xh) :
        return self.w_
    
    def w_i (self, i, t, x_xh, swap_x) :
        return self.w_

    def wh (self, t, x_xh) :
        return self.wh_
    
    def wh_i (self, i, t, x_xh, swap_x) :
        return self.wh_
    
    def cut_all (self) :
        partitions = []
        part_avg = (self.w_ + self.wh_) / 2
        w_wh = np.concatenate((self.w_, self.wh_))

        for part_i in range(2**self.w__len) :
            part = np.copy(w_wh)
            for ind in range (self.w__len) :
                part[ind + self.w__len*((part_i >> ind) % 2)] = part_avg[ind]
            partitions.append(ConstantDisturbanceIF(part[:self.w__len],part[self.w__len:],self.mode))

        # print(partitions)
        return partitions

    def __repr__(self) -> str:
        return f'ConstantDisturbanceIF(w={self.w_},wh={self.wh_})'