from ReachMM import NeuralNetworkControl, NeuralNetworkControlIF
from ReachMM import NeuralNetwork
from DoubleIntegratorModel import DoubleIntegratorModel

device = 'cuda:0'
model = '10r10r10'

nn = NeuralNetwork('models/' + model,False,device)

print('success')