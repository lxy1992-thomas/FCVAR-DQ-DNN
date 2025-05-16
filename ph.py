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

from torch import nn, Tensor
from localrq import lprq0

def ph(y,x,tau,h,u,z):
  
  n=len(y)
  p=np.size(x,1) 
  p1=p+1
  p4=2*p+2
  
  gamma=np.zeros((n,1))
  zzero=np.zeros((n,p1))
  x0=np.zeros((n,p4))
  x1=np.hstack((np.ones((n,1)),x))
  xl=np.zeros((n,p4))
  xr=np.zeros((n,p4))
  xstar=np.zeros((n,p4))
  indicater=np.zeros((n,1))
  def ker(x): return 0.75*(1-x**2)*(np.absolute(x)<=1) 
  
  q_005=np.percentile(u,5)
  q_095=np.percentile(u,95)
  
  fv, dv, RSSm, yhat=lprq0(y,x,h,tau,u,z) 

  for v in range(n):
    
     uzz=u-z[v]
    
     for k in range(n): 
       w0=(ker((uzz[k])/h)/h)+(10**-6)
       ww0=(w0)*(1 if q_005<=u[k] and u[k]<=q_095 else 0)
       dx=x1[k,]*((uzz[k])/h)
       xstar[k,]=np.hstack((x1[k,],dx))
      
       indicater[k]=(1 if y[k]<=np.dot(x1[k,],fv[k,])+(1/np.sqrt(n*h)) else 0)-(1 if y[k]<=np.dot(x1[k,],fv[k,]) else 0)
       xl[k,]=indicater[k]*(xstar[k,])
       xr[k,]=(xstar[k,])*ww0*h
  
     omegae1=(1/np.sqrt(n*h))*np.dot(xl.T,xr)
     sinv=np.linalg.inv(omegae1+(10**-7)*np.identity(p4))
     x0[v,]=np.hstack((x1[v,],zzero[v,]))
     gamma[v]=(1/(n*h))*(ker(0))*np.dot((x0[v,]),np.dot(sinv,x0[v,]))
  
  gamma=np.sum(gamma)
  
  return gamma,RSSm
    