#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 09:50:44 2018

@author: chris

@filename: 04_parameter_estimation.py
"""


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


#%% Load prior input dataset

input_data = pickle.load(open("../DATA/model_input_data.pkl","rb"))
pred_model = pickle.load(open("../DATA/pred_model_stan.pkl","rb"))

#%% Code compilation
# Stan parameter estimation using regularized marginalized MLE

gp_opt2 = pystan.StanModel(file='parameter_estimation.stan')

#%% Stan parameter estimation using regularized marginalized MLE, and data prep

#data = stan_rdump(c("N", "x", "y",
#             "N_predict", "x_predict", "y_predict",
#             "sample_idx"), file="gp.data.R")
#opt_fit = optimizing(gp_opt2, data=data, seed=5838298, hessian=FALSE)
opt_fit = gp_opt2.optimizing(data=input_data, seed=5838298)

pred_data = {'alpha':opt_fit["alpha"], 
             'rho':opt_fit["rho"], 
             'sigma':opt_fit["sigma"], 
             'N':input_data["N"],
             'x':input_data["x"],
             'y':input_data["y"],
                  'N_predict':input_data["N_predict"], 
                  'x_predict':input_data["x_predict"]}


# Alternatively, can concatenate two dicts (** is unpack dicts):
pred_data_test = {**opt_fit,**input_data}
# dictionary.pop and dictionary.get <- 

#%%
pred_opt_fit = pred_model.sampling( data=pred_data, iter=1000, warmup=0,
                    chains=4, seed=5838298, refresh=1000, algorithm="Fixed_param")


#%%


#%% Model data frame extraction and plotting

#pred_fit.plot()
pred_fit_summary = pred_opt_fit.summary(pars={"y_predict"})
pred_fit_df = pred_opt_fit.to_dataframe(diagnostics=False) # NO diagnostics for Fixed param sampling

# Still need to stack details form summary() 
#pred_fit_summary["summary_colnames"]
pred_fit_sum_df = pd.DataFrame(pred_fit_summary["summary"],columns = pred_fit_summary["summary_colnames"],)

# Plot prediction of quantiles around zeromean_daily_value
# Have lines for sampled mean, sampled median, CI for 50%, CI for 95%

#dailymean = multiyear_train_station[['dailymean','nthdayofyear']].drop_duplicates()['dailymean'] # dedupe based on index
#nthdayofyear = multiyear_train_station[['dailymean','nthdayofyear']].drop_duplicates()['nthdayofyear']
# Create the plot object
_, ax = plt.subplots()
    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
ax.plot(x, y, lw = 0.5, color = '#5380af', alpha = 1, label = 'Data')
ax.plot(x_predict,pred_fit_sum_df["50%"], lw = 2, color = '#539caf', alpha = 1, label = 'Fit')
ax.plot(x_predict,pred_fit_sum_df["mean"], lw = 2, color = '#539caf', alpha = 1, label = 'Fit')
ax.fill_between(x_predict,pred_fit_sum_df["25%"],pred_fit_sum_df["75%"] , color = '#539caf', alpha = 0.4, label = '50% CI')
ax.fill_between(x_predict,pred_fit_sum_df["2.5%"],pred_fit_sum_df["97.5%"] , color = '#539caf', alpha = 0.2, label = '95% CI')
                
                
                

_, ax = plt.subplots()
    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
ax.plot(input_data["x"], input_data["y"], lw = 0.5, color = '#5380af', alpha = 1, label = 'Data')
ax.plot(input_data["x_predict"],pred_fit_sum_df["50%"], lw = 2, color = '#539caf', alpha = 1, label = 'Fit')
ax.plot(input_data["x_predict"],pred_fit_sum_df["mean"], lw = 2, color = '#539caf', alpha = 1, label = 'Fit')
ax.fill_between(input_data["x_predict"],pred_fit_sum_df["25%"],pred_fit_sum_df["75%"] , color = '#539caf', alpha = 0.4, label = '50% CI')
ax.fill_between(input_data["x_predict"],pred_fit_sum_df["2.5%"],pred_fit_sum_df["97.5%"] , color = '#539caf', alpha = 0.2, label = '95% CI')                