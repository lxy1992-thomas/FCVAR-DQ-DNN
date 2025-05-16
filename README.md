This is the public code of the paper "A Functional-Coefficient VAR Model for Dynamic Quantiles and Its Application to Constructing Nonparametric Financial Network" by Zongwu Cai, Xiyuan Liu and Liangjun Su (2025).

To get started, you first need to install python package PyTorch (https://pytorch.org), torchtuples (https://pypi.org/project/torchtuples) and R package quantreg

1. fvdcq_dnn_simulation_main_MADE.ipynb: Jupyter file to replicate the MADE results in the simulation study in Section 3.

2. fvdcq_dnn_simulation_main_ci.ipynb: Jupyter file to replicate the AECR results in the simulation study in Section 3.

3. fvdcq_dnn_validation.ipynb: Jupyter file for choosing the best models using the validation dataset.
   
4. dqAux.py: auxiliary functions for the DPLQR.

plaqr.R: R code to implement PLAQR.

plot.R: R code to plot confidence intervals and prediction errors.

data: Concrete Compressive Strength Data Set (concrete.csv), available at https://archive.ics.uci.edu.
