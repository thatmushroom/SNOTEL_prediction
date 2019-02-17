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


#%% Temporary switch: 
# Take case of stationtriple_code == 0, 1 only
# This works. Do I have to do some sort of one-hot-ness? Nope. 0/1/2 should work. 

X = X[X.stationtriplet_codes.isin([0,1,2])]
Y = Y[Y.stationtriplet_codes.isin([0,1,2])]

#X = X[X.stationtriplet_codes.isin([0,1])] # Given a rank of 1 and output_dim of 2, this works just fine. 
#Y = Y[Y.stationtriplet_codes.isin([0,1])]

## 
#%%
#X = X.apply(pd.to_numeric)
#Y = Y.apply(pd.to_numeric)
X = X.values
Y = Y.values

#%%
#D=3
#Q=2
#R = [1, 1]
input_dim=1

#coregions = [gpflow.kernels.Coregion(input_dim, output_dim=D, rank=R[q], active_dims=[1])
#             for q in range(Q)]
#print(coregions[1])

# Per https://github.com/GPflow/GPflow/issues/542, 
# change rank of coregionalization to 2

#%% Coregionionalization 
with gpflow.defer_build():
    k_0 = gpflow.kernels.Periodic(1,active_dims=[0])
    k_0.period.prior = gpflow.priors.Gaussian(1,0.05) 
    k1 = gpflow.kernels.Matern32(1, active_dims=[0])
#    k1 =  gpflow.kernels.Matern32(1)
    input_dim=1

    coreg = gpflow.kernels.Coregion(input_dim = input_dim, output_dim = 3, 
                                    rank=2, active_dims = [1]) # Should have rank 2?
    # 2 related work just fine. 3 is when it starts to break down
#        coreg = gpflow.kernels.Coregion(input_dim = input_dim, output_dim = 2, 
#                                    rank=1, active_dims = [1])
    #coreg = gpflow.kernels.Coregion(1, output_dim=3, rank=1, active_dims=[1]) # 1 2 1 [1] for 2-param. 1 2 2 [1]? Nope. 1 3 1 [1]?
#    kern = k_0 * k1 * coreg #+ k_0 * k1 * coregions[1]
    kern = k_0 * k1 * coreg + gpflow.kernels.White(1) #+ k_0 * k1 * coregions[1]
    
    lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian()]) # One likelihood per output dim
    # TODO: 2 gives some sort of partition error. 3 gives non-invertable matrix
    m = gpflow.models.VGP(X, Y, kern=kern, likelihood=lik, num_latent=1)



m.compile()
print(kern)
#%%
#m.kern.kernels[2].W = np.random.randn(3, 2) # (output_dim, rank)?
#m.kern.kernels[2].W = np.random.randn(3, 1) # (output_dim, rank)?
m.kern.kernels[0].kernels[2].W = np.random.randn(3, 2) # (output_dim, rank)? # Used when adding white noise kernel

#m.kern.kernels[0].kernels[2].W = np.random.randn(3, 1) # (output_dim, rank)?
#m.kern.kernels[1].kernels[2].W = np.random.randn(3, 1)
print(kern)


# Thar be some memory issues. Seems to work after updating gpflow + tensorflow?
# That's because its now mostly using the CPU, not the GPU. Drat
# gpflow.train.ScipyOptimizer().minimize(model=m)

# Or, try Adam optimizer?

o = gpflow.train.AdamOptimizer(0.01)
o.minimize(m, maxiter=150) 

# gpflow.train.ScipyOptimizer().minimize(m) Just checking: Scipy optimizer doesn't work either

# And, it breaks on the cholesky decomposition.
# TODO: Fix the kernel specification? Start with only two sites, not 5
# Memory issues! Sparse -> dense doesn't play well?



# Partition error
 #partitions[896] = 2 is not in [0, 2)
	# [[node gradients_4/VGP-6a4a4d7f-160/DynamicPartition_grad/DynamicPartition (defined at C:\Anaconda3\lib\site-packages\gpflow\training\tensorflow_optimizer.py:54)  = DynamicPartition[T=DT_INT32, num_partitions=2, _device="/job:localhost/replica:0/task:0/device:CPU:0"](gradients_4/VGP-6a4a4d7f-160/DynamicPartition_grad/Reshape, VGP-6a4a4d7f-160/Cast_1)]]



#%% Coregionalization references
# https://github.com/GPflow/GPflow/issues/563 # Main gpflow reference?
# https://arxiv.org/pdf/1106.6251.pdf
# https://github.com/silburt/gp_nn/blob/master/Coregion-py3.ipynb
# http://gpss.cc/gpss17/slides/multipleOutputGPs.pdf

#%% 
# 

#def plotkernelsample(k, ax, xmin=-3, xmax=3):
#    xx = np.linspace(xmin, xmax, 100)[:,None]
#    K = k.compute_K_symm(xx)
#    ax.plot(xx, np.random.multivariate_normal(np.zeros(100), K, 3).T)
#    ax.set_title(k.__class__.__name__)
#
#def plotkernelfunction(K, ax, xmin=-3, xmax=3, other=0):
#    xx = np.linspace(xmin, xmax, 100)[:,None]
#    K = k.compute_K_symm(xx)
#    ax.plot(xx, k.compute_K(xx, np.zeros((1,1)) + other))
#    ax.set_title(k.__class__.__name__ + ' k(x, %f)'%other)
#
#plotkernelfunction(kern)
#xx = np.linspace(-2, 2, 100)[:,None]
#   K = kern.compute_K_symm(xx)


coreg.W.value @ coreg.W.value.T + np.diag(coreg.kappa.value) # Covariance matrix

Xa = multiyear_week_df[['dayssincemindate','stationtriplet_codes']]
Ya = multiyear_week_df[['value','stationtriplet_codes']]

X1 = Xa[Xa.stationtriplet_codes.isin([0])]
Y1 = Ya[Ya.stationtriplet_codes.isin([0])]
X2 = Xa[Xa.stationtriplet_codes.isin([1])]
Y2 = Ya[Ya.stationtriplet_codes.isin([1])]

X1 = X1.dayssincemindate.values
Y1 = Y1.value.values
X2 = X2.dayssincemindate.values
Y2 = Y2.value.values


def plot_gp(x, mu, var, color='k'):
    plt.plot(x, mu, color=color, lw=2)
    plt.plot(x, mu + 2*np.sqrt(var), '--', color=color)
    plt.plot(x, mu - 2*np.sqrt(var), '--', color=color)

def plot(m):
    xtest = np.linspace(X1.min(), X1.max(), 1000)[:,None]
    line, = plt.plot(X1, Y1, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.zeros_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())

    line, = plt.plot(X2, Y2, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.ones_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())

plot(m)
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
