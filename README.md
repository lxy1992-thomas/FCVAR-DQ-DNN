This is the public code of the paper "A Functional-Coefficient VAR Model for Dynamic Quantiles and Its Application to Constructing Nonparametric Financial Network" by Zongwu Cai, Xiyuan Liu and Liangjun Su (2025).

To get started, you first need to install python package PyTorch (https://pytorch.org), torchtuples (https://pypi.org/project/torchtuples) and R package quantreg. 

1. fvdcq_dnn_simulation_main_MADE.ipynb: Jupyter file to replicate the MADE results in the simulation study in Section 3.

2. fvdcq_dnn_simulation_main_ci.ipynb: Jupyter file to replicate the AECR results in the simulation study in Section 3.

3. fvdcq_dnn_validation.ipynb: Jupyter file to select the best neural network models in the simulation study using the validation dataset.
   
4. empirical_study_main.ipynb: Jupyter file to replicate the empirical results in Section 4.

5. bandwidthaic.py: Python code of AIC criterion used for selecting bandwidth h.

6. ci.py: Python code for calculating confidence interval.

7. dqAux_new.py: auxiliary functions for sparsely connected neural network.

8. localrq.py: Python code for local linear quantile estimation.

9. localrqgrid.py: Python code for local linear quantile estimation at each grid point.

10. ph.py: auxiliary function for bandwidthaic.py.

data: log difference return data used in the empirical study. The raw dataset is downloaded in Yahoo Finance and the Federal Reserve Bank of St. Louis.
