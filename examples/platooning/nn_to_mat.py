import os
import torch
import torch.nn as nn
from ReachMM.neural import NeuralNetwork
from pathlib import Path
# from numpy import identity
import numpy as np
from scipy.io import savemat

base = Path('./models/mat')

def write_mat () :
    net = NeuralNetwork('models/100r100r2')

    data = dict()
    data['act_fcns'] = []
    for layer in net.seq :
        if type(layer) == nn.ReLU :
            data['act_fcns'].append('relu')
    data['act_fcns'].append('linear')

    data['b'] = []
    data['W'] = []

    for layer in net.seq :
        if type(layer) == nn.Linear :
            W = layer.weight.cpu().detach().numpy()
            b = layer.bias.cpu().detach().numpy()

            data['W'].append(W.astype(np.float64))
            data['b'].append(b.reshape(-1,1).astype(np.float64))
    
    savemat('./models/platoon_2_100.mat', data)

if __name__ == '__main__' :
    write_mat()
