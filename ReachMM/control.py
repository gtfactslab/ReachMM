import numpy as np

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
