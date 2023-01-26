# ReachMM
## Clone the Repo and its Submodules
```
git clone --recurse-submodules https://github.com/gtfactslab/ReachMM.git
cd ReachMM
```

## Installing ReachMM into a Conda Environment
```
conda create -n ReachMM python=3.10
conda activate ReachMM
```
Install Pytorch according to [https://pytorch.org/](https://pytorch.org/). If you're using CUDA, check to make sure your CUDA version matches your nvidia driver with `nvidia-smi`.

Install `auto_LiRPA` (information taken from [https://github.com/Verified-Intelligence/auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)).
```
cd auto_LiRPA
python setup.py install
```
If you want their native CUDA modules (CUDA toolkit required),
```
python auto_LiRPA/cuda_utils.py install
```

Step back into the root folder and install the ReachMM package and its dependencies.
```
cd ..
pip install -e .
```

## Reproducing Figures from L4DC 2023 Submission

The extended version with proofs is available on [arXiv](https://arxiv.org/abs/2301.07912).

To reproduce the figures from the paper, run the following, where `runtime_N` specifies the number of runs to average over. This can take a while for large values of N.
```
cd examples/vehicle
python vehicle.py --runtime_N 1
```
```
cd examples/quadrotor
python quadrotor.py --runtime_N 1
```
