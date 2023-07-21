import os
import torch
import torch.nn as nn
from ReachMM.neural import NeuralNetwork
from pathlib import Path
# from numpy import identity

base = Path('./models/txt')

def write_txt (N) :
    n = 4*N

    # Creating NN with extra layer for states
    net = NeuralNetwork('models/100r100r2')
    net.seq.insert(0,nn.Linear(n, 4, False))
    W = torch.zeros((4, n))
    W[0,0] = 1; W[1,1] = 1; W[2,2] = 1; W[3,3] = 1
    net.seq[0].weight = nn.Parameter(W)

    name = f'platoon{N}_2_100'
    print(f'Creating {base}/{name}')

    with open(base.joinpath(name), 'w') as txt :
        def txtprint(s) :
            txt.write(f'{s}'); txt.write('\n')
        txtprint (f'{n}') # Number of inputs
        txtprint ('2') # Number of outputs
        txtprint ('3') # Number of hidden layers
        txtprint ('4') # Neurons in reduction layer
        txtprint ('100') # Neurons in first hidden layer
        txtprint ('100') # Neurons in second hidden layer

        txtprint ('Affine') # Activation function for reduction layer
        txtprint ('ReLU') # Activation function for first layer
        txtprint ('ReLU') # Activation function for second layer
        txtprint ('Affine') # Activation function for output

        for layer in net.seq :
            if type(layer) == nn.Linear :
                W = layer.weight
                b = layer.bias
                for i in range(W.shape[0]) :
                    for j in range(W.shape[1]) :
                        txtprint (W[i,j])
                # for i in range(W.shape[0]) :
                    txtprint (b[i]) if b is not None else txtprint (0)


        txtprint (0) # Offset
        txtprint (1) # Scaling


if __name__ == '__main__' :
    for N in [1,2,4,9,20,50] :
        write_txt(N)
