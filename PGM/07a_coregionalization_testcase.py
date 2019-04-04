# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:41:31 2019

@author: Chris
"""


import gpflow
import pandas as pd
import os
import numpy as np

os.chdir('Z:\SNOTEL\SNOTEL_prediction\PGM')






relpath = '../DATA/Initial_multiyear_prototype_data.csv'
#relpath_pickle = '../DATA/Initial_multiyear_prototype_data.pkl'
multiyear_df = pd.read_csv(relpath ,sep=",", engine='python')
#multiyear_df = pd.read_pickle(relpath_pickle,  compression=None)




multiyear_df['stationtriplet_codes'] = pd.Categorical(multiyear_df.stationtriplet).codes


#weekmask = multiyear_df.index % 7 == 0  
# Do based on day of week
multiyear_df.date = pd.to_datetime(multiyear_df.date)

#weekmask = multiyear_df.date.dt.dayofweek == 0
weekmask = multiyear_df.date.dt.dayofweek.isin([2,4]) # Python version of %in%
multiyear_week_df = multiyear_df[weekmask]

# Only do 2016-2018?
cutoff_date = np.datetime64('2014-08-01')

multiyear_week_df = multiyear_week_df[multiyear_week_df.date >= cutoff_date] 

# Convert "date" to "days since mindate"
datemin = min(multiyear_week_df.date)
multiyear_week_df['dayssincemindate'] = ( (multiyear_week_df.date - datemin).dt.days ) / 365.


# For coregionalization, form 2-column input/output with data in col1 and "dimension" (label) in col2
X = multiyear_week_df[['dayssincemindate','stationtriplet_codes']]
Y = multiyear_week_df[['value','stationtriplet_codes']]


#2 time series
X1 = X[X.stationtriplet_codes.isin([0,1])]
Y1 = Y[Y.stationtriplet_codes.isin([0,1])]
X1 = X1.values
Y1 = Y1.values

# 3 time series
X2 = X[X.stationtriplet_codes.isin([0,1,2])]
Y2 = Y[Y.stationtriplet_codes.isin([0,1,2])]
X2 = X2.values
Y2 = Y2.values
#%% 2 time series
# Model build: 
with gpflow.defer_build():
    k_0 = gpflow.kernels.Periodic(1)
    k_0.period.prior = gpflow.priors.Gaussian(1,0.05) 
    k1 = gpflow.kernels.Matern32(1, active_dims=[0])
    
    input_dim = 1
    output_dim = 2
    rank = 1
    coreg = gpflow.kernels.Coregion(input_dim = input_dim, output_dim = output_dim, 
                                    rank=1, active_dims = [1]) # Should have rank 2?
    # 2 output dims work just fine. 3 is when it starts to break down


    kern = k_0 * k1 * coreg + gpflow.kernels.White(1) #+ k_0 * k1 * coregions[1]
    
    lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian()]) # One likelihood per output dim

    m = gpflow.models.VGP(X1, Y1, kern=kern, likelihood=lik, num_latent=1)


m.compile()

m.kern.kernels[0].kernels[2].W = np.random.randn(output_dim, rank) # (output_dim, rank) # IF not adding the white noise kernel, drop the kernels[0].
print(kern)


#%% Minimizing 2 related time series:

o = gpflow.train.AdamOptimizer(0.01)
o.minimize(m, maxiter=150) 

#%% 3 time series
# Model build: 
with gpflow.defer_build():
    k_0_3d = gpflow.kernels.Periodic(1)
    k_0_3d.period.prior = gpflow.priors.Gaussian(1,0.05) 
    k1_3d = gpflow.kernels.Matern52(1, active_dims=[0])
    input_dim_3d = 1
    output_dim_3d = 3
    rank_3d = 2
    
    coreg_3d = gpflow.kernels.Coregion(input_dim = input_dim_3d, output_dim = output_dim_3d, 
                                    rank=rank_3d, active_dims = [1]) # Should have rank 2?
    # 2 output dims work just fine. 3 is when it starts to break down


#    kern_3d = k_0_3d * k1_3d * coreg_3d + gpflow.kernels.White(1) #+ k_0 * k1 * coregions[1]
    kern_3d = k_0_3d * coreg_3d + gpflow.kernels.White(1) #+ k_0 * k1 * coregions[1]
    
    lik_3d = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian()]) # One likelihood per output dim
    # TODO: 2 gives some sort of partition error. 3 gives non-invertable matrix
    m_3d = gpflow.models.VGP(X2, Y2, kern=kern_3d, likelihood=lik_3d, num_latent=1) 


m_3d.compile()
print(kern_3d)

#m_3d.kern.kernels[0].kernels[2].W = np.random.randn(output_dim_3d, rank_3d) # (output_dim, rank) # IF not adding the white noise kernel, drop the kernels[0].
m_3d.kern.kernels[0].kernels[1].W = np.random.randn(output_dim_3d, rank_3d) # (output_dim, rank) # IF not adding the white noise kernel, drop the kernels[0].
print(kern_3d)

#%% Minimizing 2 related time series:

o2 = gpflow.train.AdamOptimizer(0.01)
o2.minimize(m_3d, maxiter=1500) 







#%%

import matplotlib.pyplot as plt
Xa = multiyear_week_df[['dayssincemindate','stationtriplet_codes']]
Ya = multiyear_week_df[['value','stationtriplet_codes']]

X1a = Xa[Xa.stationtriplet_codes.isin([0])]
Y1a = Ya[Ya.stationtriplet_codes.isin([0])]
X2a = Xa[Xa.stationtriplet_codes.isin([1])]
Y2a = Ya[Ya.stationtriplet_codes.isin([1])]
X3a = Xa[Xa.stationtriplet_codes.isin([2])]
Y3a = Ya[Ya.stationtriplet_codes.isin([2])]

X1a = X1a.dayssincemindate.values
Y1a = Y1a.value.values
X2a = X2a.dayssincemindate.values
Y2a = Y2a.value.values
X3a = X3a.dayssincemindate.values
Y3a = Y3a.value.values

def plot_gp(x, mu, var, color='k'):
    plt.plot(x, mu, color=color, lw=2)
    plt.plot(x, mu + 2*np.sqrt(var), '--', color=color)
    plt.plot(x, mu - 2*np.sqrt(var), '--', color=color)

def plot(m):
    xtest = np.linspace(X1a.min(), X1a.max(), 1000)[:,None]
    line, = plt.plot(X1a, Y1a, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.zeros_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())

    line, = plt.plot(X2a, Y2a, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.ones_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())

#plot(m_3d)


def plot_3d(m):
    xtest = np.linspace(X1a.min(), X1a.max(), 1000)[:,None]
    line, = plt.plot(X1a, Y1a, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.zeros_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())

    line, = plt.plot(X2a, Y2a, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.ones_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())
    
    line, = plt.plot(X3a, Y3a, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.ones_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())
    

plot_3d(m_3d)
