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
import scipy.integrate as integrate
import scipy.special as special

from torch import nn, Tensor
from localrq import lprq0

def cifint(y,x,gv,h,tau,u,z):
  # y -- response variable; 
  # x -- design matrix; 
  # u -- varying variable;
  # h -- bandwidth; 
  # z -- grid point.
    
  def ker(x): return 0.75*(1-x**2)*(np.absolute(x)<=1)
  def rho(x,a): return x*(a-(1 if x<0 else 0))
  def phi(x,a): return (a-(1 if x<0 else 0))
  v0=integrate.quad(lambda x: (0.75*(1-x**2)*(np.absolute(x)<=1))**2,float('-inf'), float('inf'))[0]
  mu2=integrate.quad(lambda x: (x**2)*(0.75*(1-x**2)*(abs(x)<=1)),float('-inf'), float('inf'))[0]
  
  n=len(y)                            
  ng=len(z)
  q=np.size(x,1)
  ngrid=len(z)
  w01=np.zeros((n,1))
  oone=np.ones((n,1))
  q2=2*q+2
  q3=q+1
  q4=2*q
  q5=q3+1

  xl=np.zeros((n,q3))
  xr=np.zeros((n,q3))
  indicater2=np.zeros((n,1))
  x1=np.hstack((oone,x))
  
  bias=np.zeros((ngrid,q3))
  uerror=np.zeros((ngrid,q3))
  lerror=np.zeros((ngrid,q3))
  variance=np.zeros((ngrid,q3))
  
  for i in range(ngrid):
    
    uzz=u-z[i]
    
    for j in range(n):
      
      w01[j]=((ker((uzz[j])/h)/h)+10**-5)
      
      indicater2[j]=(1 if y[j]<=np.dot(x1[j,],gv[i,])+(1/(((n)**(1/8)))) and y[j]>np.dot(x1[j,],gv[i,])-(1/((n)**(1/8))) else 0)/(2/((n)**(1/8)))
      xl[j,]=indicater2[j]*(x1[j,])
      xr[j,]=(x1[j,])*w01[j]
  
    
    omegae1=(1/n)*np.dot(xl.T,xr)
    omegae0=(1/n)*np.dot(x1.T,xr)
    
    sinv=np.linalg.inv(omegae1+np.identity(q3)*10**(-7))
    sigmaroot=v0*tau*(1-tau)*np.dot(np.dot(sinv,omegae0),sinv)
    variance[i,]=np.diagonal(sigmaroot)
    #bias[i,]=(((h2)^2)/2)*ggv[i,]*mu2
    uerror[i,]=(norm.ppf(0.975,0,1)*np.sqrt(variance[i,])/np.sqrt(n*h))
    lerror[i,]=(norm.ppf(0.025,0,1)*np.sqrt(variance[i,])/np.sqrt(n*h))
  
  return uerror,lerror
  