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
from scipy.integrate import solve_ivp

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

dtraj, runtime = run_time(model.compute_trajectory, x0d, t_span, 0.01, embed=True, enable_bar=False)
dtt = dtraj['t']; dxx = dtraj['x']; duu = dtraj['u']
print(f"One Embedded Trajectory Time: {runtime}")

rs, runtime = run_time(model.compute_reachable_set, x0d, t_span, 0, 1, enable_bar=False)
print(runtime)
print(rs(1.25))
