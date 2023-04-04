import torch
import torch.nn as nn

def get_device(log=True) :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if log : 
        print(f"Using {device} device")
    return device

class QuadrotorNeuralNetwork(nn.Module) :
    def __init__(self, device=get_device(False)):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear( 6,32), nn.ReLU(),
            nn.Linear(32,32), nn.ReLU(),
            nn.Linear(32,3)
        )
        state_dict = torch.load('model.pt')
        self.seq.load_state_dict(state_dict())
        self.seq.eval()

    def forward(self,x):
        return self.seq(x)
    
    def __getitem__(self,idx) :
        return self.seq[idx]