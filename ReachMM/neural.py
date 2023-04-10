import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torch.utils.data import Dataset
from ReachMM.control import *
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict
import os
from ReachMM.decomp import d_metzler, d_positive


class NeuralNetwork (nn.Module) :
    def __init__(self, dir=None, load=True, device='cpu') :
        super().__init__()

        self.dir = dir
        self.mods = []
        with open(os.path.join(dir, 'arch.txt')) as f :
            arch = f.read().split()

        inputlen = int(arch[0])

        for a in arch[1:] :
            if a.isdigit() :
                self.mods.append(nn.Linear(inputlen,int(a)))
                inputlen = int(a)
            else :
                if a == 'ReLU' :
                    self.mods.append(nn.ReLU())

        self.seq = nn.Sequential(*self.mods)

        if load :
            loadpath = os.path.join(dir, 'model.pt')
            self.load_state_dict(torch.load(loadpath, map_location=device))
            print(f'Successfully loaded model from {loadpath}')

        self.device = device
        # self.dummy_input = torch.tensor([[0,0,0,0,0]], dtype=torch.float64).to(device)
        self.to(self.device)

    def forward(self, x) :
        return self.seq(x)
    
    def __getitem__(self,idx) :
        return self.seq[idx]
    
    def save(self) :
        savepath = os.path.join(self.dir, 'model.pt')
        print(f'Saving model to {savepath}')
        torch.save(self.state_dict(), savepath)

class NeuralNetworkData (Dataset) :
    def __init__(self, X, U) :
        self.X = X
        self.U = U
    def __len__(self) :
        return self.X.shape[0]
    def __getitem__(self, idx) :
        return self.X[idx,:], self.U[idx,:]
    def maxU(self) :
        maxs, maxi = torch.max(self.U, axis=0)
        return maxs

class ScaledMSELoss (nn.MSELoss) :
    def __init__(self, scale, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.scale = scale
    def __call__(self, output, target) :
        return super().__call__(output/self.scale, target/self.scale)

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

class NeuralNetworkControlIFIBP (ControlInclusionFunction) :
    def __init__(self, nn, mode='local', bound_opts=None, device='cpu', x_len=None, u_len=None, verbose=False, custom_ops=None, **kwargs):
        super().__init__(u_len=nn[-1].out_features if u_len is None else u_len, mode=mode)
        self.x_len = nn[0].in_features if x_len is None else x_len
        self.global_input = torch.zeros([1,self.x_len], dtype=torch.float32)
        self.bnn = BoundedModule(nn, self.global_input, bound_opts, device, verbose, custom_ops)
        self.method = 'IBP'
        self.device = device

    def state_transform (self, x_xh, to='numpy') :
        h = len(x_xh) // 2
        if to == 'numpy':
            x_L = np.copy(x_xh[:h])
            x_U = np.copy(x_xh[h:])
        elif to == 'torch' :
            x_L = torch.tensor(x_xh[:h].reshape(1,-1), dtype=torch.float32)
            x_U = torch.tensor(x_xh[h:].reshape(1,-1), dtype=torch.float32)
        return h, x_L, x_U

    def prime (self, x_xh) :
        pass
    
    def u (self, t, x_xh) :
        h, x, xh = self.state_transform(x_xh, to='torch')

        ptb = PerturbationLpNorm(norm=np.inf, x_L=x, x_U=xh)
        input = BoundedTensor(self.global_input, ptb)
        self.u_lb, self.u_ub = self.bnn.compute_bounds(x=(input,), method='IBP')

        return self.u_lb.cpu().detach().numpy().reshape(-1)
    
    def u_i (self, i, t, x_xh, swap_x) :
        h, x, xh = self.state_transform(x_xh, to='numpy')
        if swap_x :
            x[i] = xh[i]
        else :
            xh[i] = x[i]
        return self.u(t, np.concatenate((x,xh)))

    def uh (self, t, x_xh) :
        h, x, xh = self.state_transform(x_xh, to='torch')

        ptb = PerturbationLpNorm(norm=np.inf, x_L=x, x_U=xh)
        input = BoundedTensor(self.global_input, ptb)
        self.u_lb, self.u_ub = self.bnn.compute_bounds(x=(input,), method='IBP')

        return self.u_ub.cpu().detach().numpy().reshape(-1)

    def uh_i(self, i, t, x_xh, swap_x) :
        h, x, xh = self.state_transform(x_xh, to='numpy')
        if swap_x :
            x[i] = xh[i]
        else :
            xh[i] = x[i]
        return self.uh(t, np.concatenate((x,xh)))


class NeuralNetworkControlIF (ControlInclusionFunction) :
    def __init__(self, nn, st=None, method='CROWN', mode='hybrid', bound_opts=None, device='cpu', x_len=None, u_len=None, verbose=False, custom_ops=None, model=None, **kwargs):
        super().__init__(u_len=nn[-1].out_features if u_len is None else u_len,mode=mode)
        self.x_len = nn[0].in_features if x_len is None else x_len
        self.global_input = torch.zeros([1,self.x_len], dtype=torch.float32)
        self.st = st
        self.nn = nn
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
        
        # disclti, ltv mode
        if mode == 'disclti' or mode == 'ltv' :
            self.A = None
            self.B = None
            self.Bp = None
            self.Bn = None
            self.c = None
            self._Mm = None
            self._Mn = None
            self.M_m = None
            self.M_n = None
            self.get_ABc = None
    
    def state_transform (self, x_xh, to='numpy') :
        h = len(x_xh) // 2
        if to == 'numpy':
            x_L = np.copy(x_xh[:h])
            x_U = np.copy(x_xh[h:])
            if self.st is not None :
                x_L = self.st.numpy(x_L)
                x_U = self.st.numpy(x_U)
        elif to == 'torch' :
            x_L = torch.tensor(x_xh[:h].reshape(1,-1), dtype=torch.float32)
            x_U = torch.tensor(x_xh[h:].reshape(1,-1), dtype=torch.float32)
            if self.st is not None :
                x_L = self.st(x_L)
                x_U = self.st(x_U)
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
        self._C = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['lA'].cpu().detach().numpy().reshape(self.u_len,-1)
        self.C_ = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['uA'].cpu().detach().numpy().reshape(self.u_len,-1)
        self._d = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['lbias'].cpu().detach().numpy().reshape(-1,1)
        self.d_ = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['ubias'].cpu().detach().numpy().reshape(-1,1)
        # self._Cp = np.clip(self._C, 0, np.inf)
        # self._Cn = np.clip(self._C,-np.inf, 0)
        # self.C_p = np.clip(self.C_, 0, np.inf)
        # self.C_n = np.clip(self.C_,-np.inf, 0)
        self._Cp, self._Cn = d_positive(self._C, True)
        self.C_p, self.C_n = d_positive(self.C_, True)
        if self.mode == 'disclti' or self.mode == 'ltv' :
            # print(self.A.shape)
            # print(self.Bp.shape)
            # print(self._Cp.shape)
            # print(self._Cp[0,:])
            if self.mode == 'ltv' :
                self.A, self.B, self.c = self.get_ABc((x_xh[:h] + x_xh[h:])/2)
                self.Bp, self.Bn = d_positive(self.B, True)
            self._Mm, self._Mn = d_positive((self.A + self.Bp@self._C + self.Bn@self.C_), True)
            self.M_m, self.M_n = d_positive((self.A + self.Bp@self.C_ + self.Bn@self._C), True)

            J = jacobian(self.nn, torch.Tensor((x_xh[:h] + x_xh[h:])/2)).cpu().detach().numpy()
            Acl = self.A + self.B@J
            L, V  = np.linalg.eig(Acl)
            # print(self.A + self.B@J)
            # print(np.abs(L))
            # print(np.max(np.sum(np.abs(Acl),axis=1)))
            # print(Acl)
    
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
