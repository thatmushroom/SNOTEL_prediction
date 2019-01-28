# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:03:22 2018

@author: Chris
"""


import gpflow
import pandas as pd
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

#os.chdir('/mnt/CommunalMemory/thatmushroom/SNOTEL/SNOTEL_prediction/PGM/')
os.chdir('Z:\SNOTEL\SNOTEL_prediction\PGM')






#filepath = '/run/user/1000/gvfs/smb-share:server=192.168.0.23,share=homes/thatmushroom/SNOTEL/SNOTEL_prediction/DATA/Initial_multiyear_prototype_data.csv'
relpath = '../DATA/Initial_multiyear_prototype_data.csv'
relpath_pickle = '../DATA/Initial_multiyear_prototype_data.pkl'
multiyear_df = pd.read_csv(relpath ,sep="|", engine='python')
multiyear_df = pd.read_pickle(relpath_pickle,  compression=None)

#f = open(relpath, "r")
#test = f.read()
#f.close()
#multiyear_df = pd.read_csv("../DATA/Initial_multiyear_prototype_data.csv",
#                           sep="|", header=[0])


train_test_mask = multiyear_df['snowyear'] % 2 == 0
stats.describe(train_test_mask) # A frequency table would be better, but, eh
multiyear_train = multiyear_df[train_test_mask]
multiyear_test = multiyear_df[~train_test_mask]

# Start with predicting for one station
multiyear_train_station = multiyear_df[multiyear_df.stationtriplet == '1010:OR:SNTL']
# Plot the one station
# https://matplotlib.org/gallery/text_labels_and_annotations/date.html
# Because it tries to show all the ticks otherwise. Oh ggplot, you're so sane
import matplotlib.dates as mdates
#import matplotlib.cbook as cbook

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

_, ax = plt.subplots()
ax.plot(multiyear_train_station.date,multiyear_train_station.value)
# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years...
#datemin = np.datetime64(multiyear_train_station.date[0], 'Y')
#datemax = np.datetime64(multiyear_train_station.date[-1], 'Y') + np.timedelta64(1, 'Y')
#ax.set_xlim(datemin, datemax)
# There's other stuff, but, ignore for now

#%% Zero-mean function
# Here, do grab daily mean from all samples and subtract it out, still based on nth day of year

year_mean = np.mean(multiyear_train_station["value"])
daily_year_mean = multiyear_train_station.groupby(["nthdayofyear"]).mean()
daily_year_mean = daily_year_mean.rename(columns={"value": "dailymean",
                                                             "dataprecision":"dataprecision_y"}) 
daily_year_mean = daily_year_mean[['dataprecision_y',  'dailymean']]


multiyear_train_station = multiyear_train_station.join(daily_year_mean, on='nthdayofyear')
multiyear_train_station['zeromean_daily_value'] = multiyear_train_station['value'] -  multiyear_train_station['dailymean']
multiyear_train_station['flat_zeromean_daily_value'] = multiyear_train_station['value'] - year_mean
plt.plot(multiyear_train_station.nthdayofyear,multiyear_train_station.zeromean_daily_value)
# Join daily_year_mean (index and value)

_, ax = plt.subplots()
ax.plot(multiyear_train_station.date,multiyear_train_station.zeromean_daily_value)
# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

# Code assumes a zero-mean function and models residuals about that. 
# Therefore, find mean per day per station from training dataset and subtract that out
_, ax = plt.subplots()
ax.plot(multiyear_train_station.date,multiyear_train_station.flat_zeromean_daily_value)
# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)



#%% 2 years, sampled weekly. 7

#mask = (multiyear_train_station["snowyear"] == 2004) | (multiyear_train_station["snowyear"] == 2005)
#multiyear_train_station = multiyear_train_station[mask]
weekmask = multiyear_train_station.index % 7 == 0 
multiyear_train_station = multiyear_train_station[weekmask]
#%% 
# Two thoughts on data: Either the problem comes from multiple observations per X, 
# Or it comes from the curve going to zero variance

# Trial 1: Remove likely zero-variance!
#multiyear_train_station = multiyear_train_station[multiyear_train_station.nthdayofyear > 40]
#multiyear_train_station = multiyear_train_station[multiyear_train_station.nthdayofyear < 240]

# Convert "date" to "days since mindate
datemin = pd.to_datetime(multiyear_train_station.date[0])
(pd.to_datetime(multiyear_train_station.date) - pd.to_datetime(multiyear_train_station.date[0])).dt.days


#%%

X = np.asarray((pd.to_datetime(multiyear_train_station.date) - pd.to_datetime(multiyear_train_station.date[0])).dt.days) 
X = X.reshape(X.shape[0],1)
X = X/365.     # Scal
Y = np.asarray(multiyear_train_station['flat_zeromean_daily_value'])
Y = Y.reshape(Y.shape[0],1)

#%% Plotting
def plot(m):
    xx = np.linspace(X.min(), 1.5*X.max(), 1000).reshape(1000, 1)
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'C0', lw=2)
    plt.fill_between(xx[:,0],
                     mean[:,0] - 2*np.sqrt(var[:,0]),
                     mean[:,0] + 2*np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)
    plt.xlim(X.min(), 1.5*X.max())



#%%
plt.plot(X,Y)

#%% Matern kernel

k = gpflow.kernels.Matern52(1)
m = gpflow.models.GPR(X, Y, kern=k)

m.as_pandas_table()
gpflow.train.ScipyOptimizer().minimize(model=m)
m.as_pandas_table()

#%%
#m.build_objective()
plot(m)

#m.like
#%% 

k2 = gpflow.kernels.Matern52(1) * gpflow.kernels.Periodic(1)
m = gpflow.models.GPR(X, Y, kern=k2)

m.as_pandas_table()
gpflow.train.ScipyOptimizer().minimize(model=m)
m.as_pandas_table()

#%%
print(k2)
plot(m)


#%% 

k3 = gpflow.kernels.Periodic(1)*gpflow.kernels.Matern12(1) + gpflow.kernels.Matern12(1)
m = gpflow.models.GPR(X, Y, kern=k3)

m.as_pandas_table()
gpflow.train.ScipyOptimizer().minimize(model=m)
m.as_pandas_table()

#%%
print(k3)
plot(m)

#%%


with gpflow.defer_build():
    k_0 = gpflow.kernels.Periodic(1)
    k_0.period.prior = gpflow.priors.Gaussian(1,0.05) 
    #k_0.variance.prior = gpflow.priors.Gamma(20,20)
    k_1 = gpflow.kernels.Matern32(1)
    #k_1.lengthscales.prior = gpflow.priors.Gamma(.5,.5)
    
    k_2 = gpflow.kernels.Matern32(1)
    #k_2.lengthscales.prior = gpflow.priors.Gamma(.5,.5)
    #k_2.variance.prior = gpflow.priors.Gamma(20,20)
    
    
    k3 = k_0*k_1 + k_2
    
    
    #m = gpflow.models.GPR(X, Y, kern=k3)
    l_3 = gpflow.likelihoods.Gaussian(20)
    m = gpflow.models.VGP(X, Y, kern=k3, likelihood=l_3)
    
    # Slower, but, predictions are more accurate? Now, can i do it with a heteroskedastic ?

m.compile()
print(k3)

gpflow.train.ScipyOptimizer().minimize(model=m)

#%%
# This looks to be the winner, at least for a while. 
print(k3)
plot(m)
m.as_pandas_table()



################### END
#%%
#
#with gpflow.defer_build():
#    k_0 = gpflow.kernels.Periodic(1)
#    k_0.period.prior = gpflow.priors.Gaussian(1,0.05) 
#    k_0.period.trainable = False
#    k_0.lengthscales.prior = gpflow.priors.Gamma(0.5,1)
#    k_0.variance.prior = gpflow.priors.Gamma(17,1)
#    
#    k_1 = gpflow.kernels.Matern32(1)
#    k_1.lengthscales.prior = gpflow.priors.Gamma(1.54,1)
#    k_1.variance.prior = gpflow.priors.Gamma(17,1)
#    
#    k_2 = gpflow.kernels.Matern32(1)
#    k_2.lengthscales.prior = gpflow.priors.Gamma(.05,1)
#    k_2.variance.prior = gpflow.priors.Gamma(30,1)
#    
#    k_0_mean = gpflow.kernels.Bias(1)
#    
#    k3 = k_0_mean  + k_0*k_1 + k_2
#    #l_3 = gpflow.likelihoods.Gaussian(20)
#    m_HMC = gpflow.models.GPR(X, Y, kern=k3)
#
#m_HMC.build()
#print(m_HMC)
#
##%%  HMC
#
#sampler = gpflow.train.HMC()
#samples = sampler.sample(m_HMC, num_samples=gpflow.test_util.notebook_niter(500), epsilon=0.05, lmin=10, lmax=20, logprobs=False)
#
##%% 
#print(m_HMC)
#plot(m_HMC)
#%% Sparse VGP
#z = np.linspace(X.min() + .11,X.max()-0.1,500)
#z = z.reshape(z.shape[0],1)
#with gpflow.defer_build():
#    k_0 = gpflow.kernels.Periodic(1)
#    k_0.period.prior = gpflow.priors.Gaussian(1,0.05) 
#    k_1 = gpflow.kernels.Matern32(1)
#    k_1.lengthscales.prior = gpflow.priors.Gamma(.5,.5)
#    k_2 = gpflow.kernels.Matern32(1)
#    k_2.lengthscales.prior = gpflow.priors.Gamma(.5,.5)
#    k3 = k_0*k_1 + k_2
#    
#    #m = gpflow.models.GPR(X, Y, kern=k3)
#    m = gpflow.models.SVGP(X, Y, kern=k3, likelihood=gpflow.likelihoods.Gaussian(),
#                           Z=z)
#    
#    # Slower, but, predictions are more accurate? Now, can i do it with a heteroskedastic ?
#
#m.compile()
#print(k3)
#
#gpflow.train.ScipyOptimizer().minimize(model=m)
#
##%%
#print(k3)
#plot(m)
#



# Just, save one kernel, 
