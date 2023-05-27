import numpy as np
import interval
from interval import get_iarray, get_lu
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torch.utils.data import Dataset
from ReachMM.control import *
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict
import os
# from ReachMM.decomp import d_metzler, d_positive
import time

class NeuralNetwork (nn.Module) :
    def __init__(self, dir=None, load=True, device='cpu') :
        super().__init__()

        self.dir = dir
        self.mods = []
        self.out_len = None
        with open(os.path.join(dir, 'arch.txt')) as f :
            arch = f.read().split()

        inputlen = int(arch[0])

        for a in arch[1:] :
            if a.isdigit() :
                self.mods.append(nn.Linear(inputlen,int(a)))
                inputlen = int(a)
                self.out_len = int(a)
            else :
                if a.lower() == 'relu' :
                    self.mods.append(nn.ReLU())
                elif a.lower() == 'sigmoid' :
                    self.mods.append(nn.Sigmoid())
                elif a.lower() == 'tanh' :
                    self.mods.append(nn.Tanh())

        self.seq = nn.Sequential(*self.mods)

        if load :
            loadpath = os.path.join(dir, 'model.pt')
            self.load_state_dict(torch.load(loadpath, map_location=device))
            # print(f'Successfully loaded model from {loadpath}')

        self.device = device
        # self.dummy_input = torch.tensor([[0,0,0,0,0]], dtype=torch.float64).to(device)
        self.to(self.device)

    def forward(self, x) :
        return self.seq(x)
    
    def __getitem__(self,idx) :
        return self.seq[idx]
    
    def __str__ (self) :
        return f'neural network from {self.dir}, {str(self.seq)}'
    
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
    def __init__(self, nn, mode='hybrid', method='CROWN', bound_opts=None, device='cpu', x_len=None, u_len=None, uclip=np.interval(-np.inf,np.inf), verbose=False, custom_ops=None, model=None, **kwargs):
        super().__init__(u_len=nn.out_len if u_len is None else u_len,mode=mode)
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
        self.C_ = None
        self._Cp = None
        self._Cn = None
        self.C_p = None
        self.C_n = None
        self.C = None
        self._d = None
        self.d_ = None
        self.d = None
        self.uclip = uclip
        self._uclip, self.u_clip = get_lu(uclip)
        
    def u (self, t, x) :
        if x.dtype == np.interval :
            # Assumes .prime was called beforehand.
            # u = (self.C @ x + self.d).reshape(-1)
            _x, x_ = get_lu(x)
            _u = self._Cp @ _x + self._Cn @ x_ + self._d
            u_ = self.C_p @ x_ + self.C_n @ _x + self.d_
            # u = get_iarray(_u, u_)
            # ret_u = np.max(_u, self._uclip)
            ret_u = np.clip(_u, self._uclip, self.u_clip)
            retu_ = np.clip(u_, self._uclip, self.u_clip)
            return get_iarray(ret_u, retu_)
            # return np.intersection(u, self.uclip)
        else :
            xin = torch.tensor(x.astype(np.float32),device=self.device)
            u = self.nn(xin).cpu().detach().numpy().reshape(-1)
            return np.clip(u, self._uclip,self.u_clip)
    
    # Primes the control if to work for a range of x_xh (finds _C, C_, _d, d_)
    def prime (self, x) :
        if x.dtype != np.interval :
            return
            # raise Exception('Call prime with an interval array')
        _x, x_ = get_lu(x)
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
        self._Cp, self._Cn = d_positive(self._C, True)
        self.C_p, self.C_n = d_positive(self.C_, True)
        self.C = get_iarray(self._C, self.C_)
        self._d = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['lbias'].cpu().detach().numpy().reshape(-1)
        self.d_ = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['ubias'].cpu().detach().numpy().reshape(-1)
        self.d = get_iarray(self._d, self.d_)
        # print('\n_C', self._C)
        # print('C_', self.C_)
        # print('_d', self._d)
        # print('d_', self.d_)
    
    def __str__(self) -> str:
        return f'{str(self.nn)}, {self.mode} mode'
