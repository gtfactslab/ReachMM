import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from ReachMM import ControlFunction
from VehicleUtils import file_to_numpy

# For training purposes only, run this file to train.
MODELNAME = 'twoobs'
FILENAMES = ['twoobs_20221101-181935','twoobs_20221101-190818', 'twoobs_20221101-194609', 'twoobs_20221101-231318']
# FILENAMES = ['twoobs_20221101-181935']
EPOCHS = 50
LEARNING = 0.2
LEARNING_GAM = 0.9
TRAINP = 0.9
LAYER1 = 100
LAYER2 = 100

def get_device(log=True) :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if log : 
        print(f"Using {device} device")
    return device

class VehicleStateTransformation (nn.Module) :
    def __init__(self):
        super().__init__()
        self.in_features = 4
        self.out_features = 5
    def forward (self,x) :
        x = torch.atleast_2d(x)
        return torch.stack([x[:,0], x[:,1], torch.sin(x[:,2]), torch.cos(x[:,2]), x[:,3]], dim=1)
    def numpy (self,x) :
        return np.array([x[0],x[1],np.sin(x[2]),np.cos(x[2]),x[3]])
    def __repr__(self):
        return f'VehicleStateTransformation(in_features={self.in_features},out_features={self.out_features})'

class VehicleNeuralNetwork (nn.Module) :
    def __init__(self, file=None, device=get_device(False)) :
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(4,LAYER1,dtype=torch.float32), nn.ReLU(),
            nn.Linear(LAYER1,LAYER2,dtype=torch.float32), nn.ReLU(),
            nn.Linear(LAYER2,2,dtype=torch.float32)
        )
        if file != None :
            loadpath = 'models/' + file + '.pt'
            print(f'Loading model from {loadpath}')
            self.load_state_dict(torch.load(loadpath))
        self.device = device
        # self.dummy_input = torch.tensor([[0,0,0,0,0]], dtype=torch.float64).to(device)
        self.to(self.device)
    def forward(self, x) :
        return self.seq(x)
    
    def __getitem__(self,idx) :
        return self.seq[idx]
    # def u(self,t,x) :
    #     # u = self(VehicleNNController.npstate_to_nninput(x).to(self.device))\
    #     #             .cpu().detach().numpy().reshape(-1)
    #     u = self(torch.tensor(x,device=self.device)).cpu().detach().numpy().reshape(-1)
    #     u[0] = np.clip(u[0],-20,20)
    #     u[1] = np.clip(u[1],-np.pi/3,np.pi/3)
    #     return u

class VehicleData (Dataset) :
    def __init__(self, X, U) :
        # self.X = torch.stack([X[:,0], X[:,1], torch.sin(X[:,2]), torch.cos(X[:,2]), X[:,3]], dim=1)
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

# Train the Neural Network
if __name__ == '__main__' :
    device = get_device()

    X, U = file_to_numpy(FILENAMES)
    SPLIT_LOC = int(TRAINP * X.shape[0])

    Xtrain = X[:SPLIT_LOC,:].astype(np.float32)
    Utrain = U[:SPLIT_LOC,:].astype(np.float32)
    Xtest = X[SPLIT_LOC:,:].astype(np.float32)
    Utest = U[SPLIT_LOC:,:].astype(np.float32)

    print(Xtrain.dtype)

    fig, axs = plt.subplots(2, 2,figsize=[8,8], dpi=100)
    axs = axs.reshape(-1)
    axs[0].scatter(Xtest[:,0], Xtest[:,1], s=1)
    axs[0].set_title("y vs x")
    axs[0].set_xlim([-15,15]); axs[0].set_ylim([-15,15])
    axs[1].scatter(Xtest[:,2], Xtest[:,3], s=1)
    axs[1].set_title("v vs psi")
    axs[2].scatter(Utest[:,0], Utest[:,1], s=1)
    axs[2].set_title("uang vs uacc (data)")
    plt.ion(); plt.show(); plt.pause(1)
    # plt.draw()

    Xtrain = torch.from_numpy(Xtrain).to(device)
    Utrain = torch.from_numpy(Utrain).to(device)
    Xtest = torch.from_numpy(Xtest).to(device)
    Utest = torch.from_numpy(Utest).to(device)

    train_data = VehicleData(Xtrain, Utrain)
    test_data  = VehicleData(Xtest, Utest)
    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dl   = DataLoader(test_data, batch_size=256, shuffle=True)
    test_dl  = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    print("Training set size: ", len(train_data))
    print("Testing  set size: ", len(test_data))

    st = VehicleStateTransformation().to(device)
    net = VehicleNeuralNetwork().to(device)
    print(st)
    print(net)

    learning_rate = LEARNING
    loss = ScaledMSELoss(scale=train_data.maxU())
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LEARNING_GAM)

    Xt, Ut = next(iter(test_dl))
    for epoch in range(EPOCHS) :
        pbar = tqdm(iter(train_dl))
        for Xi, Ui in pbar :
            l = loss(net(st(Xi)), Ui)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            pbar.set_description(f'Epoch {epoch+1}, Loss={l:.4f}')
        scheduler.step()
        Upred = net(st(Xt))
        test_loss = loss(Upred, Ut)
        Upred = Upred.detach().cpu().numpy()
        axs[3].clear()
        axs[3].scatter(Upred[:,0], Upred[:,1], s=1)
        axs[3].set_xlim(axs[2].get_xlim()); axs[3].set_ylim(axs[2].get_ylim())
        axs[3].set_title("uang vs uacc (predicted)")
        fig.suptitle(f"Scaled RMSE on Testing Set: {torch.sqrt(test_loss)}.4")
        print(f'Epoch: {epoch+1}, RMSE={torch.sqrt(test_loss)}:4f')
        plt.ion(); plt.show(); plt.pause(0.2)
    
    print(f'Loss on Full Test Set: {test_loss}')

    savepath = 'models/' + MODELNAME + '.pt'
    print(f'Saving to {savepath}')
    torch.save(net.state_dict(), savepath)

    plt.ioff(); plt.show()
