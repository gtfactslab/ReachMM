# ReachMM

## Conda Setup
```
conda create -n ReachMM python=3.10
conda activate ReachMM
```
Install Pytorch according to [https://pytorch.org/](https://pytorch.org/). If you're using CUDA, check to make sure your CUDA version matches your nvidia driver with ` nvidia-smi`.

Install `auto_LiRPA` (information taken from [https://github.com/Verified-Intelligence/auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)).
```
cd auto_LiRPA
python setup.py install
```
If you want their native CUDA modules (CUDA toolkit required),
```
python auto_LiRPA/cuda_utils.py install
```

Next, install the ReachMM package and its dependencies.
```
pip install -e .
```