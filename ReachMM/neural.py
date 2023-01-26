import numpy as np
import torch
import torch.nn as nn
from ReachMM.control import *
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict

class NeuralNetworkControl (ControlFunction) :
    def __init__(self, nn:nn.Module, st=None, device='cpu', u_len=None):
        super().__init__(u_len=nn[-1].out_features if u_len is None else u_len)
        self.nn = nn
        self.st = st
        self.device = device

    def u(self, t, x) :
        if self.st is not None :
            xin = self.st(torch.tensor(x.astype(np.float32),device=self.device))
        else :
            xin = torch.tensor(x.astype(np.float32),device=self.device)
        u = self.nn(xin).cpu().detach().numpy().reshape(-1)
        return u

class NeuralNetworkControlIF (ControlInclusionFunction) :
    def __init__(self, nn, st=None, method='CROWN', mode='hybrid', bound_opts=None, device='cpu', x_len=None, u_len=None, verbose=False, custom_ops=None, **kwargs):
        super().__init__(u_len=nn[-1].out_features if u_len is None else u_len,mode=mode)
        self.x_len = nn[0].in_features if x_len is None else x_len
        self.global_input = torch.zeros([1,self.x_len], dtype=torch.float32)
        self.st = st
        self.bnn = BoundedModule(nn, self.global_input, bound_opts, device, verbose, custom_ops)
        # self.global_input = global_input
        self.method = method
        self.device = device
        self.required_A = defaultdict(set)
        self.required_A[self.bnn.output_name[0]].add(self.bnn.input_name[0])
        self._C = None
        self._Cp = None
        self._Cn = None
        self.C_ = None
        self.C_p = None
        self.C_n = None
        self._d = None
        self.d_ = None
        self.u_lb = None
        self.u_ub = None
    
    def state_transform (self, x_xh, to='numpy') :
        h = len(x_xh) // 2
        if to == 'numpy':
            x_L = np.copy(x_xh[:h])
            x_U = np.copy(x_xh[h:])
            if self.st is not None :
                x_L = self.st.numpy(x_L)
                x_U = self.st.numpy(x_U)
            for i in range(h) :
                flat_x_L = x_L.reshape(-1)
                flat_x_U = x_U.reshape(-1)
                # if flat_x_U[i] < flat_x_L[i] :
                #     print(f'swapping {i}')
                #     temp = np.copy(flat_x_L[i])
                #     flat_x_L[i] = flat_x_U[i]
                #     flat_x_U[i] = temp
        elif to == 'torch' :
            x_L = torch.tensor(x_xh[:h].reshape(1,-1), dtype=torch.float32)
            x_U = torch.tensor(x_xh[h:].reshape(1,-1), dtype=torch.float32)
            if self.st is not None :
                x_L = self.st(x_L)
                x_U = self.st(x_U)
            for i in range(h) :
                flat_x_L = x_L.reshape(-1)
                flat_x_U = x_U.reshape(-1)
                # if flat_x_U[i] < flat_x_L[i] :
                #     print(f'swapping {i}')
                #     temp = torch.clone(flat_x_L[i])
                #     flat_x_L[i] = flat_x_U[i]
                #     flat_x_U[i] = temp
        return h, x_L, x_U


    # Primes the control if to work for a range of x_xh (finds _C, C_, _d, d_)
    def prime (self, x_xh) :
        h, x_L, x_U = self.state_transform(x_xh, to='torch')
        # x_L = torch.tensor([x_xh[:h]], dtype=torch.float32)
        # x_U = torch.tensor([x_xh[h:]], dtype=torch.float32)
        # print(x_L)
        ptb = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
        input = BoundedTensor(self.global_input, ptb)
        self.u_lb, self.u_ub, A_dict = \
            self.bnn.compute_bounds(x=(input,), method=self.method, return_A=True, needed_A_dict=self.required_A)
        self.u_lb = self.u_lb.cpu().detach().numpy()
        self.u_ub = self.u_ub.cpu().detach().numpy()
        self._C = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['lA'].cpu().detach().numpy()
        self.C_ = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['uA'].cpu().detach().numpy()
        self._d = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['lbias'].cpu().detach().numpy().reshape(-1,1)
        self.d_ = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['ubias'].cpu().detach().numpy().reshape(-1,1)
        self._Cp = np.clip(self._C, 0, np.inf)
        self._Cn = np.clip(self._C,-np.inf, 0)
        self.C_p = np.clip(self.C_, 0, np.inf)
        self.C_n = np.clip(self.C_,-np.inf, 0)
    
    def u (self, t, x_xh) :
        h, x, xh = self.state_transform(x_xh, to='numpy')
        return (self._Cp @ x.reshape(-1,1) + self._Cn @ xh.reshape(-1,1) + self._d).reshape(-1)
    
    def u_i (self, i, t, x_xh, swap_x) :
        h = len(x_xh) // 2
        h, x, xh = self.state_transform(x_xh, to='numpy')
        if swap_x :
            x[i] = xh[i]
        else :
            xh[i] = x[i]
        x_xh_ = np.concatenate((x,xh))
        if self.mode == 'local' :
            self.prime(x_xh_)
        return (self._Cp @ x.reshape(-1,1) + self._Cn @ xh.reshape(-1,1) + self._d).reshape(-1)

    def uh(self, t, x_xh) :
        h, x, xh = self.state_transform(x_xh, to='numpy')
        return (self.C_p @ xh.reshape(-1,1) + self.C_n @ x.reshape(-1,1) + self.d_).reshape(-1)

    def uh_i(self, i, t, x_xh, swap_x) :
        h, x, xh = self.state_transform(x_xh, to='numpy')
        if swap_x :
            x[i] = xh[i]
        else :
            xh[i] = x[i]
        x_xh_ = np.concatenate((x,xh))
        if self.mode == 'local' :
            self.prime(x_xh_)
        return (self.C_p @ xh.reshape(-1,1) + self.C_n @ x.reshape(-1,1) + self.d_).reshape(-1)
