import os
import torch
import torch.nn as nn
from ReachMM.neural import NeuralNetwork
from pathlib import Path
from numpy import identity

base = Path('./models')
model_dirs = [d for d in base.iterdir() if d.is_dir()]

for model_dir in model_dirs :
    print(f'\nParsing model in {model_dir}')
    model_name = model_dir.relative_to(base)
    with open(model_dir.joinpath(model_name), 'r') as txt :
        txt_read = txt.read().split()

    # Making arch.txt
    print(f'Creating {model_dir}/arch.txt')
    with open(model_dir.joinpath('arch.txt'), 'w') as arch :
        arch.write(txt_read[0] + ' ') # number of inputs
        num_layers = int(txt_read[2])
        # hidden layers
        for layer_i in range(num_layers) :
            arch.write(txt_read[3 + layer_i] + ' ')
            arch.write(txt_read[3 + num_layers + layer_i] + ' ')
        # number of outputs
        arch.write(txt_read[1] + ' ')
        # extra activation
        arch.write(txt_read[3 + num_layers + num_layers] + ' ')
        # extra affine layer
        arch.write(txt_read[1])

    idx = 3 + num_layers + num_layers + 1

    # Making a NeuralNetwork object, without any state dict loaded
    net = NeuralNetwork(model_dir, load=False)

    for layer in net.seq[:-1] :
        if type(layer) == nn.Linear :
            W = torch.empty_like(layer.weight)
            b = torch.empty_like(layer.bias)
            for i in range(W.shape[0]) :
                for j in range(W.shape[1]) :
                    W[i,j] = float(txt_read[idx])
                    idx += 1
            # for i in range(b.shape[0]) :
                b[i] = float(txt_read[idx])
                idx += 1
            layer.weight = nn.Parameter(W)
            layer.bias = nn.Parameter(b)

    # Last affine layer 
    layer = net.seq[-1]
    tmp = torch.ones_like(layer.bias)
    tmp *= -1*float(txt_read[idx])
    idx += 1
    layer.bias = nn.Parameter(tmp.reshape(layer.bias.shape))
    # tmp = torch.identity(layer.weight.shape).reshape(-1)
    tmp = torch.tensor(identity(int(txt_read[1])))
    tmp *= float(txt_read[idx])
    idx += 1
    layer.weight = nn.Parameter(tmp.reshape(layer.weight.shape))

    net.save()
