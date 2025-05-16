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


def lprq0g(y,x,h,tau,u,z):

  n=len(y)                            # number of data points
  ng=len(z)
  p=np.size(x,1)
    
  yin=robjects.FloatVector(y)
  uin=robjects.FloatVector(u)
  zin=robjects.FloatVector(z)
  xin=robjects.r.matrix(robjects.FloatVector(np.reshape(x.T,(n*p,1))), ncol=p)

  robjects.r.assign('tau',tau)
  robjects.r.assign('h',h)
  robjects.r.assign('yin',yin)
  robjects.r.assign('xin',xin)
  robjects.r.assign('uin',uin)
  robjects.r.assign('zin',zin)

  robjects.r('''
  lprq0g=function(y,x,h,tau,u,z){
  # y -- response variable; 
  # x -- design matrix; 
  # u -- varying variable;
  # h -- bandwidth; 
  # z -- grid point.
  require(quantreg)
  
  ker=function(x){0.75*(1-x^2)*(abs(x)<=1)}   
  
  rho=function(x,a){x*(a-(ifelse(x<0,1,0)))}
  phi=function(x,a){(a-(ifelse(x<0,1,0)))}
  
  ngrid=length(z)                        # number of grid points
  n=length(y)                            # number of data points
  p=dim(x)[2] 
  one=matrix(1,n,1)
  p2=2*p+2
  p3=p+1
  p4=2*p
  p5=p3+1
  
  dx=matrix(0,n,p3)
  ddx=matrix(0,n,p3)
  w0=as.vector(matrix(0,n,1))
  ww0=as.vector(matrix(0,n,1))
  beta=matrix(0,ngrid,p2)
  x1=cbind(one,x)
  
  q_0.05=quantile(u,0.05,na.rm=T)
  q_0.95=quantile(u,0.95,na.rm=T)
  q_0.05=q_0.05[1]
  q_0.95=q_0.95[1]
  
  for (w in 1:ngrid){
    uzz=u-(one%*%z[w])
    
    for (j in 1:n){
      
      w0[j]=(ker((uzz[j])/h)/h)+10^-6
      dx[j,]=x1[j,]*(uzz[j])
    }
    r=rq(y~x+dx, tau=tau, weights=w0)
    beta[w,]=r$coef[1:p2]
  }  
  
  fv=beta[,1:p3]
  dv=beta[,p5:p2]
  
  finalresults=list()
  finalresults$fv=fv
  finalresults$dv=dv
  finalresults$w0=w0
  
  return(finalresults)
  
  }
  ''')  
  lprq0g=robjects.r['lprq0g']

  fv, dv, w0=lprq0g(yin,xin,h,tau,uin,zin)
  fv=np.array(fv)
  dv=np.array(dv)
  w0=np.array(w0)

  return fv, dv, w0