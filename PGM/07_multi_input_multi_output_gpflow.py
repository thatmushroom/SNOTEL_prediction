# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 10:21:04 2019

@author: Chris
"""

# 07_multi_input_multi_output_gpflow.py
# 
# https://gpflow.readthedocs.io/en/develop/notebooks/coreg_demo.html


import gpflow
import pandas as pd
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
#import plotnine 
import seaborn as sns

#os.chdir('/mnt/CommunalMemory/thatmushroom/SNOTEL/SNOTEL_prediction/PGM/')
os.chdir('Z:\SNOTEL\SNOTEL_prediction\PGM')






#filepath = '/run/user/1000/gvfs/smb-share:server=192.168.0.23,share=homes/thatmushroom/SNOTEL/SNOTEL_prediction/DATA/Initial_multiyear_prototype_data.csv'
relpath = '../DATA/Initial_multiyear_prototype_data.csv'
relpath_pickle = '../DATA/Initial_multiyear_prototype_data.pkl'
#multiyear_df = pd.read_csv(relpath ,sep="|", engine='python')
multiyear_df = pd.read_pickle(relpath_pickle,  compression=None)






#%% Plot 5 unique stations on one plot
multiyear_df.stationtriplet.unique()

#groups = multiyear_df.groupby('stationtriplet')

#fig=plt.figure()
#for name, group in groups:
#    plt.plot(group.date, group.value, label=name)    
##ax.legend()
#
#plt.show()



#%% Data prep:
#  Convert stationtriplet to numeric
# Can also do this using pd.factorize()[0] or sklearn.LabelEncoder
multiyear_df['stationtriplet_codes'] = pd.Categorical(multiyear_df.stationtriplet).codes
# Weekly sampling, cut to every other year


#weekmask = multiyear_df.index % 7 == 0  # Doesn't exactly work with groups. 
# Do based on day of week?
multiyear_df.date = pd.to_datetime(multiyear_df.date)

weekmask = multiyear_df.date.dt.dayofweek == 0
multiyear_week_df = multiyear_df[weekmask]
#
#
#train_test_mask = multiyear_week_df['snowyear'] % 2 == 0
#stats.describe(train_test_mask) # A frequency table would be better, but, eh
#multiyear_train = multiyear_week_df[train_test_mask]
#multiyear_test = multiyear_week_df[~train_test_mask]

sns.relplot(x='date', y='value', hue = 'stationtriplet',  kind = 'line', data=multiyear_week_df ) # Egads, this is damn slow on full dataset. 
sns.relplot(x='date', y='value', hue = 'stationtriplet_codes',  kind = 'line', data=multiyear_week_df ) # Egads, this is damn slow on full dataset. 


# Convert "date" to "days since mindate"
datemin = min(multiyear_week_df.date)
multiyear_week_df['dayssincemindate'] = ( (multiyear_week_df.date - datemin).dt.days ) / 365.


# For coregionalization, form 2-column input/output with data in col1 and "dimension" (label) in col2



X = multiyear_week_df[['dayssincemindate','stationtriplet_codes']]
Y = multiyear_week_df[['value','stationtriplet_codes']]

#X = X.apply(pd.to_numeric)
#Y = Y.apply(pd.to_numeric)
X = X.values
Y = Y.values
#%% Coregionionalization 
with gpflow.defer_build():
    k_0 = gpflow.kernels.Periodic(1)
    k_0.period.prior = gpflow.priors.Gaussian(1,0.05) 
    k1 = gpflow.kernels.Matern32(1, active_dims=[0])
    coreg = gpflow.kernels.Coregion(1, output_dim=2, rank=1, active_dims=[1])
    kern = k_0 * k1 * coreg
    
    lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(), 
                                                 gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian()]) # 2? 5? I think 5?
    m = gpflow.models.VGP(X, Y, kern=kern, likelihood=lik, num_latent=1)



m.compile()
m.kern.kernels[1].W = np.random.randn(2, 1)
print(kern)

gpflow.train.ScipyOptimizer().minimize(model=m)
# And, it breaks on the cholesky decomposition.
# TODO: Fix the kernel specification? Start with only two sites, not 5

#%% 
# 
print(k3)
plot(m)
m.as_pandas_table()


#%% Kernel notes from 06

#
#with gpflow.defer_build():
#    k_0 = gpflow.kernels.Periodic(1)
#    k_0.period.prior = gpflow.priors.Gaussian(1,0.05) 
#    k_1 = gpflow.kernels.Matern32(1)
#    k_2 = gpflow.kernels.Matern32(1)
#    
#    
#    k3 = k_0*k_1 + k_2
#    
#    l_3 = gpflow.likelihoods.Gaussian(20)
#    m = gpflow.models.VGP(X, Y, kern=k3, likelihood=l_3)
#    
#m.compile()
#print(k3)
#
#gpflow.train.ScipyOptimizer().minimize(model=m)
#
#
#print(k3)
#plot(m)
#m.as_pandas_table()



#%% Coregionalization kernel example:
# make a dataset with two outputs, correlated, heavy-tail noise. One has more noise than the other.
X1 = np.random.rand(100, 1)
X2 = np.random.rand(50, 1) * 0.5
Y1 = np.sin(6*X1) + np.random.standard_t(3, X1.shape)*0.03
Y2 = np.sin(6*X2+ 0.7) + np.random.standard_t(3, X2.shape)*0.1

plt.plot(X1, Y1, 'x', mew=2)
plt.plot(X2, Y2, 'x', mew=2);

########
# a Coregionalization kernel. The base kernel is Matern 3/2, and acts on the first ([0]) data dimension.
# the 'Coregion' kernel indexes the outputs, and acts on the second ([1]) data dimension
k1 = gpflow.kernels.Matern32(1, active_dims=[0])
coreg = gpflow.kernels.Coregion(1, output_dim=2, rank=1, active_dims=[1])
kern = k1 * coreg


# build a variational model. This likelihood switches between Student-T noise with different variances:
lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.StudentT(), gpflow.likelihoods.StudentT()])


# Augment the time data with ones or zeros to indicate the required output dimension
X_augmented = np.vstack((np.hstack((X1, np.zeros_like(X1))), np.hstack((X2, np.ones_like(X2)))))
## IE, one dimension per factor group

# Augment the Y data to indicate which likelihood we should use
Y_augmented = np.vstack((np.hstack((Y1, np.zeros_like(X1))), np.hstack((Y2, np.ones_like(X2)))))

m = gpflow.models.VGP(X_augmented, Y_augmented, kern=kern, likelihood=lik, num_latent=1)
from gpflow.test_util import notebook_niter
m.kern.kernels[1].W = np.random.randn(2, 1)

gpflow.train.ScipyOptimizer().minimize(m, maxiter=notebook_niter(2000))


def plot_gp(x, mu, var, color='k'):
    plt.plot(x, mu, color=color, lw=2)
    plt.plot(x, mu + 2*np.sqrt(var), '--', color=color)
    plt.plot(x, mu - 2*np.sqrt(var), '--', color=color)

def plot(m):
    xtest = np.linspace(0, 1, 100)[:,None]
    line, = plt.plot(X1, Y1, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.zeros_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())

    line, = plt.plot(X2, Y2, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.ones_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())

plot(m)
