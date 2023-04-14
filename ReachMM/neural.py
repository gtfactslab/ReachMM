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

class NeuralNetworkControl (Control) :
    def __init__(self, nn, st=None, method='CROWN', mode='hybrid', bound_opts=None, device='cpu', x_len=None, u_len=None, verbose=False, custom_ops=None, model=None, **kwargs):
        super().__init__(u_len=nn[-1].out_features if u_len is None else u_len,mode=mode)
        self.x_len = nn[0].in_features if x_len is None else x_len
        self.global_input = torch.zeros([1,self.x_len], dtype=torch.float32)
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
        
    def u (self, t, x) :
        xin = torch.tensor(x.astype(np.float32),device=self.device)
        u = self.nn(xin).cpu().detach().numpy().reshape(-1)
        return u
    
    # Primes the control if to work for a range of x_xh (finds _C, C_, _d, d_)
    def prime (self, _x, x_) :
        x_L = torch.tensor(_x.reshape(1,-1), dtype=torch.float32)
        x_U = torch.tensor(x_.reshape(1,-1), dtype=torch.float32)
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
        self._Cp, self._Cn = d_positive(self._C, True)
        self.C_p, self.C_n = d_positive(self.C_, True)

        # if self.mode == 'disclti' or self.mode == 'ltv' :
        #     if self.mode == 'ltv' :
        #         self.A, self.B, self.c = self.get_ABc((x_xh[:h] + x_xh[h:])/2)
        #         self.Bp, self.Bn = d_positive(self.B, True)
        #     self._Mm, self._Mn = d_positive((self.A + self.Bp@self._C + self.Bn@self.C_), True)
        #     self.M_m, self.M_n = d_positive((self.A + self.Bp@self.C_ + self.Bn@self._C), True)

        #     J = jacobian(self.nn, torch.Tensor((x_xh[:h] + x_xh[h:])/2)).cpu().detach().numpy()
        #     Acl = self.A + self.B@J
        #     L, V  = np.linalg.eig(Acl)
    
    def _u (self, t, _x, x_) :
        return (self._Cp @ _x.reshape(-1,1) + self._Cn @ x_.reshape(-1,1) + self._d).reshape(-1)

    def u_ (self, t, _x, x_) :
        return (self.C_p @ x_.reshape(-1,1) + self.C_n @ _x.reshape(-1,1) + self.d_).reshape(-1)
