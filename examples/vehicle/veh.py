import numpy as np
from ReachMM import NeuralNetworkControl, NeuralNetworkControlIF
from VehicleNeuralNetwork import VehicleNeuralNetwork
from VehicleModel import VehicleModel
from VehicleUtils import *
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from tabulate import tabulate
from ReachMM.utils import run_time

device = 'cpu'

nn = VehicleNeuralNetwork('twoobs-nost', device)

control = NeuralNetworkControl(nn, device=device)
controlif = NeuralNetworkControlIF(nn, mode='hybrid', method='CROWN', device=device)
model = VehicleModel(control, controlif, u_step=0.25)
x0 = np.array([8,8,-2*np.pi/3,2])
eps = np.array([0.1,0.1,0.01,0.01])
# t_step = 0.01 if args.t_step is None else args.t_step
xlen = 4

t_span = [0,1.25]

x0d = np.concatenate((x0 - eps,x0 + eps))

rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, 0, 0, enable_bar=True)
print(runtime)
print(rs.sol(1))