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
    if 'network' in keys :
        mat = mat['network']
        matW = mat['W'][()]
        matb = mat['b'][()]
        layer_acts = mat['activation_fcns'][()]
        layer_sizes = mat['layer_sizes'][()]
        ninputs = mat['number_of_inputs'][()]
        nlayers = mat['number_of_layers'][()]
        noutputs = mat['number_of_outputs'][()]
    else :
        matW = mat['W']
        matb = mat['b']
        layer_acts = mat['activation_fcns']
        layer_sizes = mat['layer_sizes']
        ninputs = mat['number_of_inputs']
        nlayers = mat['number_of_layers']
        noutputs = mat['number_of_outputs']

    model_dir = base_dir.joinpath('models',mat_file.with_suffix('').name)
    model_dir.mkdir(exist_ok=True)

    # Making arch.txt
    print(f'Creating {model_dir}/arch.txt')
    with open(model_dir.joinpath('arch.txt'), 'w') as arch :
        # Number of inputs
        arch.write(str(ninputs) + ' ')
        for layer_i in range(nlayers) :
            act = str(layer_acts[layer_i]).strip()
            if act.lower() != 'linear' :
                arch.write(str(layer_sizes[layer_i]) + ' ')
                arch.write(act + ' ')
        # Number of outputs
        arch.write(str(noutputs))
    
    net = NeuralNetwork(model_dir, load=False)
    
    ind = 0
    for layer in net.seq :
        if type(layer) == nn.Linear :
            W = np.copy(matW[ind]).reshape(layer.weight.shape)
            b = np.copy(matb[ind]).reshape(layer.bias.shape)
            layer.weight = nn.Parameter(torch.tensor(W))
            layer.bias = nn.Parameter(torch.tensor(b))
            ind += 1
    print(net)

    net.save()