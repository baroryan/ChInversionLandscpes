# Chi-Space Inversion for Detachment-Limited Landscapes

<p align="center">
  <img src="figures/graphical_abstract.png" width="750">
</p>

<p align="center">
  <em>Graphical overview of the chi-space inversion workflow for detachment-limited landscapes.</em>
</p>

This repository contains the codebase for performing inversions of detachment-limited landscapes in chi space. Example workflows will be uploaded shortly.

The methods are based on:

**Oryan et al. (2025)**  
https://doi.org/10.1029/2024JB030819

## Installation

This code requires `numba`, `daggerpy`, `pyscabbard`, `numpy`, `matplotlib`, `xarray`, and `seaborn`.

We highly recommend installing the code in a clean Conda environment:

```bash
export PYTHONNOUSERSITE=1
unset PYTHONPATH

conda create -n invertchi -c conda-forge python=3.11 xtensor-python pip
conda activate invertchi

pip install taichi
pip install numba
pip install daggerpy
pip install pyscabbard
pip install numpy matplotlib xarray seaborn
pip install "setuptools<81" --force-reinstall 
```

<sub>Tested with Python `3.11.15`.</sub>