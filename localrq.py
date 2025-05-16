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
import os
os.environ["R_HOME"] = f"{os.environ['CONDA_PREFIX']}\\Lib\\R"
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
importr('quantreg')

from torch import nn, Tensor


def lprq0(y,x,h,tau,u,z):
  # y -- response variable; 
  # x -- design matrix; 
  # u -- varying variable;
  # h -- bandwidth; 
  # z -- grid point.
    
  def ker(x): return 0.75*(1-x**2)*(np.absolute(x)<=1)
  def rho(x,a): return x*(a-(1 if x<0 else 0))
  def phi(x,a): return (a-(1 if x<0 else 0))
  
  n=len(y)                            # number of data points
  ng=len(z)
  p=np.size(x,1)
  p2=2*p+2
  p3=p+1
  p4=2*p
  p5=p3+1
  
  dx=np.zeros((n,p3))
  w0=np.zeros((n,1))
  ww0=np.zeros((n,1))
  beta=np.zeros((ng,p2))
  yhat=np.zeros((ng,1))
  inRSSm=np.zeros((ng,1))
  x1=np.hstack((np.ones((n,1)),x))
  
  q_005=np.percentile(u,5)
  q_095=np.percentile(u,95)
  
  for w in range(ng):
    
    uzz=u-z[w]
    
    for j in range(n):
      
      w0[j]=(ker((uzz[j])/h)/h)+(10**-6)
      ww0[j]=(w0[j])*(1 if q_005<=u[j] and u[j]<=q_095 else 0)
      dx[j,]=x1[j,]*(uzz[j])

    ww0r=robjects.FloatVector(ww0)
    yr=robjects.FloatVector(y)
    xr=robjects.r.matrix(robjects.FloatVector(np.reshape(x.T,(n*p,1))), ncol=p)
    dxr=robjects.r.matrix(robjects.FloatVector(np.reshape(dx.T,(n*p3,1))), ncol=p3)

    robjects.r.assign('tau',tau)
    robjects.r.assign('ww0r',ww0r)
    robjects.r.assign('yr',yr)
    robjects.r.assign('xr',xr)
    robjects.r.assign('dxr',dxr)

    robjects.r('''
    r=rq(yr~xr+dxr,tau=tau,weights=ww0r)
    betain=r$coef
    ''')  
      
    beta[w,:]=np.array(robjects.r['betain'])
    yhat[w]=np.dot(x1[w,:],beta[w,0:p3])+np.dot(dx[w,:],beta[w,(p5-1):p2])
    inRSSm[w]=rho(y[w]-yhat[w],tau)
    
  fv=beta[:,0:p3]
  dv=beta[:,(p5-1):p2]

  RSSm=np.mean(inRSSm)
    
  return fv, dv, RSSm, yhat
    