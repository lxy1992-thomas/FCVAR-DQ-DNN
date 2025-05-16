import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import torch
import math
import random
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from torchtuples import Model
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch import nn, Tensor
import os
os.environ["R_HOME"] = f"{os.environ['CONDA_PREFIX']}\\Lib\\R"
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
importr('quantreg')

from localrq import lprq0
from ph import ph


def bandwidthaic(y,x,tau,h0,u,z):
  
  def rho(x,a): return x*(a-(1 if x<0 else 0))

  n=len(y)                                           
  
  gamma,RSSm=ph(y,x,tau,h0,u,z)
  phv=gamma
  phe=2*(phv+1)/(n-(phv+2))
  aic=np.log(RSSm)+phe
  
  return aic, RSSm
