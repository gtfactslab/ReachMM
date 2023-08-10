import torch
from ReachMM import NeuralNetwork

net = NeuralNetwork('100r100r2')
dummy_input = torch.zeros(4)

torch.onnx.export(net.seq, dummy_input, "/home/grits/Documents/sherlock/network_files/100r100r2.onnx", 
                  verbose=True, input_names=["tensor_input"], output_names=["tensor_output"])
