#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:30:37 2018

@author: chris
@filename: 03_data_estimation.py

"""


# Take the same basic code as 02, but shape the data to be one time series.





import pandas as pd
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pystan 
import pickle

os.environ['STAN_NUM_THREADS'] = "8"

#os.chdir('Z:\SNOTEL\SNOTEL_prediction\PGM')
os.chdir('/mnt/CommunalMemory/thatmushroom/SNOTEL/SNOTEL_prediction/PGM/')

#%%

# The basic approach is to build an estimate for GP for each site individually
# Then, add correlation

# https://betanalpha.github.io/assets/case_studies/gp_part1/part1.html


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


#%% 
# Two thoughts on data: Either the problem comes from multiple observations per X, 
# Or it comes from the curve going to zero variance

# Trial 1: Remove likely zero-variance!
#multiyear_train_station = multiyear_train_station[multiyear_train_station.nthdayofyear > 40]
#multiyear_train_station = multiyear_train_station[multiyear_train_station.nthdayofyear < 240]

# Convert "date" to "days since mindate
datemin = pd.to_datetime(multiyear_train_station.date[0])
(pd.to_datetime(multiyear_train_station.date) - pd.to_datetime(multiyear_train_station.date[0])).dt.days
#%% Model data assembly 

# xvar is day of year from snowyeardate, yvar is value. Fit individually for site
# alpha is hyper-p for marginal SD, rho is length scale, sigma is measurement variability
# 

## Stan fit:
# Hyperparameters
alpha_true = 5
rho_true = 7
sigma_true = 10
# Input parameters 
N = multiyear_train_station.shape[0]
x = np.asarray((pd.to_datetime(multiyear_train_station.date) - pd.to_datetime(multiyear_train_station.date[0])).dt.days) 
# date instead of nth day of year. Convert to numeric? Really, just convert to days since original observation

y = np.asarray(multiyear_train_station['zeromean_daily_value'])
# Prediction parameters
x_predict = np.unique((pd.to_datetime(multiyear_train_station.date) - pd.to_datetime(multiyear_train_station.date[0])).dt.days) # Again, date isntead of nth day of year
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
#%% Model compile, skip if possible

pred_model = pystan.StanModel(file='./predict_gauss.stan')

#%% Model run


pred_fit = pred_model.sampling(data=onestation_data,iter=1000,chains=4,
                          algorithm="Fixed_param")

# 10syears is taking a very long time. Degeneracy? Lots of data? TBD

#%% Model data frame extraction and plotting

#pred_fit.plot()
pred_fit_summary = pred_fit.summary(pars={"y_predict"})
pred_fit_df = pred_fit.to_dataframe(diagnostics=False) # NO diagnostics for Fixed param sampling

# Still need to stack details form summary() 
#pred_fit_summary["summary_colnames"]
pred_fit_sum_df = pd.DataFrame(pred_fit_summary["summary"],columns = pred_fit_summary["summary_colnames"],)

# Plot prediction of quantiles around zeromean_daily_value
# Have lines for sampled mean, sampled median, CI for 50%, CI for 95%

dailymean = multiyear_train_station[['dailymean','nthdayofyear']].drop_duplicates()['dailymean'] # dedupe based on index
nthdayofyear = multiyear_train_station[['dailymean','nthdayofyear']].drop_duplicates()['nthdayofyear']
# Create the plot object
_, ax = plt.subplots()
    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
ax.plot(x, y, lw = 0.5, color = '#5380af', alpha = 1, label = 'Data')
ax.plot(nthdayofyear,pred_fit_sum_df["50%"], lw = 2, color = '#539caf', alpha = 1, label = 'Fit')
ax.plot(nthdayofyear,pred_fit_sum_df["mean"], lw = 2, color = '#539caf', alpha = 1, label = 'Fit')
ax.fill_between(nthdayofyear,pred_fit_sum_df["25%"],pred_fit_sum_df["75%"] , color = '#539caf', alpha = 0.4, label = '50% CI')
ax.fill_between(nthdayofyear,pred_fit_sum_df["2.5%"],pred_fit_sum_df["97.5%"] , color = '#539caf', alpha = 0.2, label = '95% CI')
