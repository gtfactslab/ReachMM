# import argparse

# parser = argparse.ArgumentParser(description="Convert mat files to ReachMM NN format")

import torch
import torch.nn as nn
from scipy.io import loadmat
import numpy as np
from ReachMM.neural import NeuralNetwork
from pathlib import Path

base_dir = Path('.')

mat_dir = base_dir.joinpath('models','mat')
for mat_file in mat_dir.iterdir() :
    print(f'\nFound {mat_file}')
    mat = loadmat(mat_file, squeeze_me=True)
    keys = [*mat]
    matW = mat['W']; matW = [np.atleast_2d(W) for W in matW]
    matb = mat['b']
    matacts = mat['act_fcns']
    # matacts[-1] = ''

    model_dir = base_dir.joinpath('models',mat_file.with_suffix('').name)
    model_dir.mkdir(exist_ok=True)

    # Making arch.txt
    print(f'Creating {model_dir}/arch.txt')
    with open(model_dir.joinpath('arch.txt'), 'w') as arch :
        # Number of inputs
        arch.write(str(matW[0].shape[1]) + ' ')
        for layer_i in range(len(matW)) :
            act = str(matacts[layer_i]).strip()
            if act.lower() != 'linear' :
                arch.write(str(matW[layer_i].shape[0]) + ' ')
                arch.write(act + ' ')
        # Number of outputs
        arch.write(str(matW[-1].shape[0]))
    
    net = NeuralNetwork(model_dir, load=False)
    print(net)
    
    ind = 0
    for layer in net.seq :
        if type(layer) == nn.Linear :
            if ind != len(matW) :
                W = np.copy(matW[ind]).reshape(layer.weight.shape)
                b = np.copy(matb[ind]).reshape(layer.bias.shape)
                layer.weight = nn.Parameter(torch.tensor(W))
                layer.bias = nn.Parameter(torch.tensor(b))
                ind += 1
            else :
                layer.weight = nn.Parameter(torch.tensor(np.array([[1.0,0],[0,1.0]])))
                layer.bias = nn.Parameter(torch.tensor(np.array([-20.0,-20.0])))
    print(net)

    net.save()
