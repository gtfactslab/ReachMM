import os
import torch
import torch.nn as nn
from ModuleFromTxt import ModuleFromTxt
from ReachMM.neural import NeuralNetwork
from pathlib import Path

base = Path('./models')
model_dirs = [d for d in base.iterdir() if d.is_dir()]
# model_dirs = [Path('nn_1_relu')]

for model_dir in model_dirs[:1] :
    print(f'Parsing model in {model_dir}')
    model_name = model_dir.relative_to(base)
    with open(model_dir.joinpath(model_name), 'r') as txt :
        txt_read = txt.read().split()

    with open(model_dir.joinpath('arch.txt'), 'w') as arch :
        arch.write(txt_read[0] + ' ')

# for subdir in subdirs :
#     models_path = os.path.join(subdir, 'models')
#     print('Looking for directories in', models_path)

#     model_dirs = os.listdir(models_path) if os.path.isdir(models_path) else []

#     for model_dir in model_dirs :
#         full_model_path = os.path.join(models_path, model_dir)
#         print('\n============')
#         print('Converting', full_model_path)
#         txt_model = ModuleFromTxt(os.path.join(full_model_path, model_dir))
#         print('Txt Model Loaded.')

#         mods = []
        
#         with open(os.path.join(full_model_path, 'arch.txt'), 'wt') as file :
#             file.write(f'{txt_model.layers[0].in_features} ')
#             for layer in txt_model.layers :
#                 if type(layer) is nn.Linear :
#                     file.write(f'{layer.out_features} ')
#                 elif type(layer) is nn.ReLU :
#                     file.write('ReLU ')
#                 elif type(layer) is nn.Sigmoid :
#                     file.write('Sigmoid ')
#                 elif type(layer) is nn.Tanh :
#                     file.write('Tanh ')
#                 mods.append(layer)
        
#         # print(txt_model)
        
#         txt_model.seq = nn.Sequential(*mods)
#         del(txt_model.layers)

#         # print(txt_model)

#         nn_model = NeuralNetwork(full_model_path, load=False)
#         print(nn_model)
#         nn_model.load_state_dict(txt_model.state_dict())
#         nn_model.save()
