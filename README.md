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

## Reproducing Figures from CDC 2023 Submission

### Vehicle Model

```
cd examples/vehicle
```

To reproduce Figure 3 for the vehicle model, run the following:
```
python cdc2023.py
```
and to reproduce Table I, run the following:
```
python cdc2023.py --table -N 10
```
where `N` specifies the number of runs to average over. This can take a while for large values of `N`.

### Double Integrator Model

```
cd examples/doubleintegrator
```

To reproduce Figures 4 and 5 for the double integrator model, and the rows for `ReachMM` and `ReachMM-CG` for Table II, run the following:
```
python cdc2023.py -N 1
```
where `N` specifies the number of runs to average over. This can take a while for large values of `N`. To reproduce the tree on Figure 2, run the following:
```
python cdc2023.py --tree
```

<!-- To reproduce the figures from the paper, run the following, where `{model}` is replaced with either `doubleintegrator` , `runtime_N` specifies the number of runs to average over. This can take a while for large values of N. -->

<!-- ## Reproducing Figures from L4DC 2023 Submission

The extended version with proofs is available on [arXiv](https://arxiv.org/abs/2301.07912).

To reproduce the figures from the paper, run the following, where `runtime_N` specifies the number of runs to average over. This can take a while for large values of N.
```
cd examples/vehicle
python vehicle.py --runtime_N 1
```
```
cd examples/quadrotor
python quadrotor.py --runtime_N 1
``` -->
