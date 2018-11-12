# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 11:39:41 2018

@author: Chris
@filename: 02_data_ingestion.py
"""


import pandas as pd
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pystan 

os.environ['STAN_NUM_THREADS'] = "8"

os.chdir('Z:\SNOTEL\SNOTEL_prediction\PGM')


#%%

# The basic approach is to build an estimate for GP for each site individually
# Then, add correlation

# https://betanalpha.github.io/assets/case_studies/gp_part1/part1.html

# Take even-numbered snow years as model fit, odd-numbered snow years as 


multiyear_df = pd.read_csv("../DATA/Initial_multiyear_prototype_data.csv")

train_test_mask = multiyear_df['snowyear'] % 2 == 0
stats.describe(train_test_mask) # A frequency table would be better, but, eh
multiyear_train = multiyear_df[train_test_mask]
multiyear_test = multiyear_df[~train_test_mask]

# Start with predicting for one station
multiyear_train_station = multiyear_train[multiyear_train.stationtriplet == '1010:OR:SNTL']
plt.plot(multiyear_train_station.nthdayofyear,multiyear_train_station.value)



#%% Zero-mean function

year_mean = np.mean(multiyear_train_station["value"])
daily_year_mean = multiyear_train_station.groupby(["nthdayofyear"]).mean()
daily_year_mean = daily_year_mean.rename(columns={"value": "dailymean",
                                                             "dataprecision":"dataprecision_y"}) 
daily_year_mean = daily_year_mean[['dataprecision_y',  'dailymean']]


multiyear_train_station = multiyear_train_station.join(daily_year_mean, on='nthdayofyear')
multiyear_train_station['zeromean_daily_value'] = multiyear_train_station['value'] -  multiyear_train_station['dailymean']
plt.plot(multiyear_train_station.nthdayofyear,multiyear_train_station.zeromean_daily_value)
# Join daily_year_mean (index and value)

# Code assumes a zero-mean function and models residuals about that. 
# Therefore, find mean per day per station from training dataset and subtract that out


#%% 
# Two thoughts on data: Either the problem comes from multiple observations per X, 
# Or it comes from the curve going to zero variance

# Trial 1: Remove likely zero-variance!
multiyear_train_station = multiyear_train_station[multiyear_train_station.nthdayofyear > 40]
multiyear_train_station = multiyear_train_station[multiyear_train_station.nthdayofyear < 240]


#%% Model data assembly 

# xvar is day of year from snowyeardate, yvar is value. Fit individually for site
# alpha is hyper-p for marginal SD, rho is length scale, sigma is measurement variability
# 

## Stan fit:
# Hyperparameters
alpha_true = 20
rho_true = 7
sigma_true = 5
# Input parameters 
N = multiyear_train_station.shape[0]
x = np.asarray(multiyear_train_station['nthdayofyear'])
y = np.asarray(multiyear_train_station['zeromean_daily_value'])
# Prediction parameters
x_predict = np.unique(multiyear_train_station['nthdayofyear'])
N_predict = x_predict.shape[0]

onestation_data = {'alpha': alpha_true,
      'rho': rho_true,
      'sigma': sigma_true,
      'N': N,
      'x': x,
      'y': y,
      'N_predict': N_predict,
      'x_predict': x_predict
      }
#%% Model run

pred_fit = pystan.stan(file='./predict_gauss.stan', data=onestation_data, iter=1000, warmup=0,
                 chains=4, seed=5838298, refresh=1000, algorithm="Fixed_param")

# If I want to avoid recompiling the model each time...
#pred_model = pystan.StanModel(file='./predict_gauss.stan')
#fit = pred_model.sampling(data=onestation_data,iter=1000,chains=4,
#                          algorithm="Fixed_param",refresh=10)

pred_fit.plot()

# Plot prediction of quantiles around zeromean_daily_value
y_total = pred_fit.extract("y_predict")['y_predict']

# TODO still:
# Plot the posterior quantiles around zeromean_daily_value from y_total.


# http://natelemoine.com/fast-gaussian-process-models-in-stan/ 
# Need to speed up processing significantly down the road?

