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

weekmask = multiyear_df.date.dt.dayofweek == 0
multiyear_week_df = multiyear_df[weekmask]



# Convert "date" to "days since mindate"
datemin = min(multiyear_week_df.date)
multiyear_week_df['dayssincemindate'] = ( (multiyear_week_df.date - datemin).dt.days ) / 365.


# For coregionalization, form 2-column input/output with data in col1 and "dimension" (label) in col2
X = multiyear_week_df[['dayssincemindate','stationtriplet_codes']]
Y = multiyear_week_df[['value','stationtriplet_codes']]



X = X[X.stationtriplet_codes.isin([0,1,2])]
Y = Y[Y.stationtriplet_codes.isin([0,1,2])]

X = X.values
Y = Y.values

# Model build: 
with gpflow.defer_build():
    k_0 = gpflow.kernels.Periodic(1)
    k_0.period.prior = gpflow.priors.Gaussian(1,0.05) 
    k1 = gpflow.kernels.Matern32(1, active_dims=[0])
    input_dim=1

    coreg = gpflow.kernels.Coregion(input_dim = input_dim, output_dim = 3, 
                                    rank=1, active_dims = [1]) # Should have rank 2?
    # 2 output dims work just fine. 3 is when it starts to break down


    kern = k_0 * k1 * coreg + gpflow.kernels.White(1) #+ k_0 * k1 * coregions[1]
    
    lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian()]) # One likelihood per output dim
    # TODO: 2 gives some sort of partition error. 3 gives non-invertable matrix
    m = gpflow.models.VGP(X, Y, kern=kern, likelihood=lik, num_latent=1)


m.compile()
print(kern)

m.kern.kernels[0].kernels[2].W = np.random.randn(3, 1) # (output_dim, rank) # IF not adding the white noise kernel, drop the kernels[0].


#%%


o = gpflow.train.AdamOptimizer(0.01)
o.minimize(m, maxiter=150) 

# gpflow.train.ScipyOptimizer().minimize(m)