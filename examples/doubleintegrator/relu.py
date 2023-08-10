import numpy as np
import interval
from interval import from_cent_pert
import sympy as sp

from ReachMM import DiscreteTimeSpec, ContinuousTimeSpec
from ReachMM import System, NeuralNetwork, NNCSystem, NeuralNetworkControl
from ReachMM import UniformPartitioner, CGPartitioner
from ReachMM.utils import run_times
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import shapely.geometry as sg
import shapely.ops as so
import polytope
import torch

net = NeuralNetwork('models/relu', False)
print(net)

net.seq[0].weight = torch.nn.Parameter(torch.tensor([[1],[-1]],dtype=torch.float32))
net.seq[0].bias = torch.nn.Parameter(torch.tensor([0,0],dtype=torch.float32))
net.seq[2].weight = torch.nn.Parameter(torch.tensor([ [1, -1] ],dtype=torch.float32))
net.seq[2].bias = torch.nn.Parameter(torch.tensor([0],dtype=torch.float32))

print(net(torch.tensor([-1],dtype=torch.float32)))

net.save()

x0 = np.array([
    np.interval(-1,2)
])

nnc = NeuralNetworkControl(net)
nnc.prime(x0)
