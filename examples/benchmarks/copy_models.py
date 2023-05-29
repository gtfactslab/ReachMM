import os
import shutil
from pathlib import Path

polar_dir = '/home/akash/Research/POLAR_Tool/examples'

for i in range(1,7) :
    bmpath = os.path.join(polar_dir, f'benchmark{i}')
    models = [d for d in os.listdir(bmpath) \
            if d[:2] == 'nn' and d[-5:] != 'crown' and d.find('.') == -1]
    for model in models :
        dst = os.path.join(f'benchmark{i}/models',model)
        os.makedirs(dst,exist_ok=True)
        shutil.copyfile(os.path.join(polar_dir, f'benchmark{i}', model), \
                        os.path.join(dst, model))
    # Path(f'benchmark{i}/benchmark{i}.py').touch()
