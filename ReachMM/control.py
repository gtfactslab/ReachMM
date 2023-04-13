import numpy as np
from ReachMM.decomp import d_positive

# ControlFunction implements a piecewise constant controller.
class ControlFunction :
    def __init__(self, u_len, mode='hybrid') :
        self.u_len = u_len
        # joint or element
        # global, hybrid, or local
        self.mode = mode
        self.uCALC  = None
        self._uCALC = None
        self.u_CALC = None
        self._uCALC_x   = None
        self.u_CALC_x  = None
        self._uCALCx_  = None
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
            if self._uCALC_x is None :
                d = len(x_)
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
            return self._uCALC_x, self.u_CALC_x, self._uCALCx_, self.u_CALC_x_

    def __call__(self, t, x) : 
        return self.step(t,x)

class LinearControl (ControlFunction) :
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

# class DisturbanceFunction :
#     def __init__(self, w_len) :
#         self.w__len = w_len
    
#     def w  (self, t, x) :
#         pass

#     def __call__(self, t, x) : 
#         return self.w_(t,x)

# class DisturbanceInclusionFunction :
#     def __init__(self, w_len, mode='hybrid') :
#         self.w__len = w_len
#         # joint or element
#         # global, hybrid, or local
#         self.mode = mode
    
#     def w  (self, t, x_xh) :
#         pass

#     def w_i (self, i, t, x_xh, swap_x) :
#         pass

#     def wh (self, t, x_xh) :
#         pass

#     def wh_i (self, i, t, x_xh, swap_x) :
#         pass

# class NoDisturbance (DisturbanceFunction) :
#     def __init__(self, w_len=0):
#         super().__init__(w_len)

#     def w  (self, t, x_xh) :
#         return np.zeros((self.w__len))

#     def wh (self, t, x_xh) :
#         return np.zeros((self.w__len))

# class NoDisturbanceIF (DisturbanceInclusionFunction) :
#     def __init__(self, w_len=0):
#         super().__init__(w_len)

#     def w  (self, t, x_xh) :
#         return np.zeros((self.w__len))

#     def w_i (self, i, t, x_xh, swap_x) :
#         return np.zeros((self.w__len))

#     def wh (self, t, x_xh) :
#         return np.zeros((self.w__len))

#     def wh_i (self, i, t, x_xh, swap_x) :
#         return np.zeros((self.w__len))

# class ConstantDisturbance (DisturbanceFunction) :
#     def __init__(self, w):
#         w = np.asarray(w)
#         super().__init__(len(w))
#         self.w_  = w

#     def w (self, t, x) :
#         return self.w_

# class ConstantDisturbanceIF (DisturbanceInclusionFunction) :
#     def __init__(self, w, wh, mode='hybrid'):
#         w = np.asarray(w)
#         wh = np.asarray(wh)
#         super().__init__(len(w), mode)
#         self.w_  = w
#         self.wh_ = wh

#     def w (self, t, x_xh) :
#         return self.w_
    
#     def w_i (self, i, t, x_xh, swap_x) :
#         return self.w_

#     def wh (self, t, x_xh) :
#         return self.wh_
    
#     def wh_i (self, i, t, x_xh, swap_x) :
#         return self.wh_
    
#     def cut_all (self) :
#         partitions = []
#         part_avg = (self.w_ + self.wh_) / 2
#         w_wh = np.concatenate((self.w_, self.wh_))

#         for part_i in range(2**self.w__len) :
#             part = np.copy(w_wh)
#             for ind in range (self.w__len) :
#                 part[ind + self.w__len*((part_i >> ind) % 2)] = part_avg[ind]
#             partitions.append(ConstantDisturbanceIF(part[:self.w__len],part[self.w__len:],self.mode))

#         # print(partitions)
#         return partitions

#     def __repr__(self) -> str:
#         return f'ConstantDisturbanceIF(w={self.w_},wh={self.wh_})'