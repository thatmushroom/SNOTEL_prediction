#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 19:27:19 2018

@author: chris
@filename: 05_kernel_fitting_gpy.py
"""

# Try building a kernel using GPY rather than stan. GPflowOpt seems to have 
# bayesian hyperparameter optimization, but, one thing at a time

#%% Package loads

os.chdir('/mnt/CommunalMemory/thatmushroom/SNOTEL/SNOTEL_prediction/PGM/')


import GPy


import pandas as pd
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pickle


#%% Copy-paste data transformation from 03

#%%

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


#%% End of copy-paste data transformation

#%% Build GP through GPY

# References
# http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb
# http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/GPyCrashCourse.ipynb

# Toy data 
X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05
#


kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) # Square exponential
kernel.plot()
X = np.asarray((pd.to_datetime(multiyear_train_station.date) - pd.to_datetime(multiyear_train_station.date[0])).dt.days) 
X = X.reshape(X.shape[0],1)
X = X/365.*2*np.pi     # Scal
Y = np.asarray(multiyear_train_station['flat_zeromean_daily_value'])
Y = Y.reshape(Y.shape[0],1)

m = GPy.models.GPRegression(X,Y,kernel)
print(m)
fig = m.plot()
#from IPython.display import display
#display(m)


#GPy.plotting.show(fig) #, filename='basic_gp_regression_notebook')

# Estimate the parameters for reals
m.optimize(messages=True)
print(m)
fig = m.plot()
# Explore parameter space randomly
m.optimize_restarts(num_restarts = 15)
print(m)
fig = m.plot()


#%% Periodic Matern32


#kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) # Square exponential and periodic?
kernel = GPy.kern.PeriodicMatern32()
kernel.plot()
m = GPy.models.GPRegression(X,Y,kernel)
print(m)
fig = m.plot()


# Estimate the parameters for reals
m.optimize(messages=True)
print(m)
fig = m.plot()

# Explore parameter space randomly
m.optimize_restarts(num_restarts = 15)
print(m)
fig = m.plot()
# Plot the kernel after estimating parameters
kernel.plot()

#%%Periodic Matern52

#kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) # Square exponential and periodic?
kernel = GPy.kern.PeriodicMatern52() # Seems like I should be able to specify the period
m = GPy.models.GPRegression(X,Y,kernel)
print(m)
fig = m.plot()


# Estimate the parameters for reals
m.optimize(messages=True)
print(m)
fig = m.plot()

# Explore parameter space randomly
m.optimize_restarts(num_restarts = 15)
print(m)
fig = m.plot()

#%% Not necessarily getting happy period choices, IE, on the order of a week.
# Try building manually ? or, PeriodicExponential?
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)  # Square exponential

kernel = GPy.kern.MLP(input_dim = 1) # MLP is step-function-ish
kernel.plot()

kernel = GPy.kern.StdPeriodic(input_dim=1)
kernel = GPy.kern.StdPeriodic(input_dim=1)*GPy.kern.RBF(input_dim=1)
kernel.plot()

#kernel = GPy.kern.StdPeriodic(input_dim=1) + GPy.kern.RBF(input_dim=1)
kernel = GPy.kern.StdPeriodic(input_dim=1)*GPy.kern.RBF(input_dim=1)*GPy.kern.MLP(input_dim = 1,   ) 
kernel.plot()


#%% Try based on locally periodic (SE*periodic)
#kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) # Square exponential and periodic?

kernel = GPy.kern.StdPeriodic(input_dim=1,lengthscale=1)*GPy.kern.RBF(input_dim=1,lengthscale=0.2)
m = GPy.models.GPRegression(X,Y,kernel)
print(m)
fig = m.plot()
kernel.plot() 

# Estimate the parameters for reals
m.optimize(messages=True)
print(m)
fig = m.plot()
kernel.plot() 

# Explore parameter space randomly
m.optimize_restarts(num_restarts = 15)
print(m)
fig = m.plot()
kernel.plot() # Spiky! But prone to overfitting to the data

#%% Locally, v2
#kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) # Square exponential and periodic?
kernel = GPy.kern.PeriodicMatern52(input_dim=1)*GPy.kern.RBF(input_dim=1)
m = GPy.models.GPRegression(X,Y,kernel)
print(m)
fig = m.plot()
kernel.plot() 

# Estimate the parameters for reals
m.optimize(messages=True)
print(m)
fig = m.plot()
kernel.plot() 

# Explore parameter space randomly
m.optimize_restarts(num_restarts = 15)
print(m)
fig = m.plot()
kernel.plot() 


#%% Locally, v3. Closer, but need to do some sort of mean-reversion for each year
#kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) # Square exponential and periodic?
kernel = GPy.kern.PeriodicMatern52(input_dim=1)+GPy.kern.RBF(input_dim=1)
m = GPy.models.GPRegression(X,Y,kernel)
print(m)
fig = m.plot()
kernel.plot() 

# Estimate the parameters for reals
m.optimize(messages=True)
print(m)
fig = m.plot()
kernel.plot() 

## Explore parameter space randomly
#m.optimize_restarts(num_restarts = 15)
#print(m)
#fig = m.plot()
#kernel.plot() 

kernel = GPy.kern.PeriodicMatern52(input_dim=1,variance=8349.6,lengthscale=0.00550617764108134,period=7.97827144248)+GPy.kern.RBF(input_dim=1,variance=115,lengthscale=0.052161)


#Name : GP regression
#Objective : 2201.7777071142827
#Number of Parameters : 6
#Number of Optimization Parameters : 6
#Updates : True
#Parameters:
#  GP_regression.                     |                 value  |  constraints  |  priors
#  sum.periodic_Matern52.variance     |      8349.60107534421  |      +ve      |        
#  sum.periodic_Matern52.lengthscale  |   0.00550617764108134  |      +ve      |        
#  sum.periodic_Matern52.period       |     7.978271442484868  |      +ve      |        
#  sum.rbf.variance                   |    114.99609174305171  |      +ve      |        
#  sum.rbf.lengthscale                |  0.052161097201822465  |      +ve      |        
#  Gaussian_noise.variance            |      6.77951547394751  |      +ve      |        


#%% spectral mixture...actually works amazingly well with 3 sets. Less well with 5.
rbf = GPy.kern.RBF(input_dim=1)
cos = GPy.kern.Cosine(input_dim=1)

k = rbf * cos + rbf * cos + rbf * cos 
m = GPy.models.GPRegression(X,Y,k)
print(m)
fig = m.plot()
k.plot() 

# Estimate the parameters for reals
m.optimize(messages=True)
print(m)
fig = m.plot()
k.plot() 

m.optimize_restarts(num_restarts = 15)
print(m)
fig = m.plot()
kernel.plot() 

#%% Heteroskedasticity! Maybe need to change to a simpler kernel, rbv+periodicmatern52 worked well
k = rbf * cos + rbf * cos + rbf * cos 
m_H = GPy.models.GPHeteroscedasticRegression(X,Y,k)
print(m_H)
fig = m_H.plot()
k.plot() 

# Estimate the parameters for reals
m_H.optimize(messages=True)
print(m_H)
fig = m_H.plot()
k.plot() 

m_H.optimize_restarts(num_restarts = 15)
print(m_H)
fig = m_H.plot()
kernel.plot() 

#%% Heteroskedasticity! Maybe need to change to a simpler kernel, rbv+periodicmatern52 worked well
k = GPy.kern.PeriodicMatern52(input_dim=1)+GPy.kern.RBF(input_dim=1)
m_H = GPy.models.GPHeteroscedasticRegression(X,Y,k)
print(m_H)
fig = m_H.plot()
k.plot() 

# Estimate the parameters for reals
m_H.optimize(messages=True)
print(m_H)
fig = m_H.plot()
k.plot() 

m_H.optimize_restarts(num_restarts = 15)
print(m_H)
fig = m_H.plot()
kernel.plot() 


 #%% Locally, v3. Closer, but need to do some sort of mean-reversion for each year
##kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) # Square exponential and periodic?
#kernel = (GPy.kern.PeriodicMatern52(input_dim=1)+GPy.kern.RBF(input_dim=1))+GPy.kern.RBF(input_dim=1, lengthscale=2  )
#m = GPy.models.GPRegression(X,Y,kernel)
#print(m)
#fig = m.plot()
#kernel.plot() 
#
## Estimate the parameters for reals
#m.optimize(messages=True)
#print(m)
#fig = m.plot()
#kernel.plot() 
#
## Explore parameter space randomly
#m.optimize_restarts(num_restarts = 15)
#print(m)
#fig = m.plot()
#kernel.plot() 

#%% How well does it perform for online prediction each year?

#%% Think about desired features for the kernel:
# Periodic, > 0 always. 
# Regularly "periodic" in that it goes to 0 and resets each year, but is not sinusoidal, nor is the maximum in the same spot regularly
# Quasi-periodic may be a suitable way to describe it
 
# Also, maximum is not in the same place each year

# In online version, need to consider current year as a noisy realization of overall trend
