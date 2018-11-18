# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:58:24 2018

@author: Chris
@filename: 01_data_ingestion.py
"""

# File for data ingestion for small SNOTEL subset to build out workflow
# TODO: Actually save the dataframe created


# Gaussian process links:
# http://www.robots.ox.ac.uk/~sjrob/Pubs/philTransA_2012.pdf - Multi-dimensional signals!
# https://betanalpha.github.io/assets/case_studies/gp_part1/part1.html 
# http://mc-stan.org/events/stancon2017-notebooks/stancon2017-trangucci-hierarchical-gps.pdf

# SNOTEL
# Daily historic data:
# https://wcc.sc.egov.usda.gov/nwcc/rgrpt?report=swe&state=CO&operation=View
# Doing an advanced search is how to pluck multiple stations at a time
# Use the Create/Modify Report tool



## Data!
# Web query
# https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/303:CO:SNTL%7Cstate=%22CO%22%20AND%20county=%22Boulder%22,%22Costilla%22,%22Eagle%22,%22Garfield%22,%22Gilpin%22,%22Grand%22,%22Gunnison%22,%22Kit%20Carson%22,%22Ouray%22,%22Saguache%22,%22San%20Juan%22,%22Summit%22%20AND%20network=%22SNTL%22%20AND%20outServiceDate=%222100-01-01%22%7Cname/POR_BEGIN,POR_END/stationId,name,WTEQ::value?fitToScreen=false

# Or, R package! Damn. R package.
# https://rhlee12.github.io/RNRCS/
#install.packages("RNRCS")

# 
# dotR <- file.path(Sys.getenv("HOME"), ".R")
# if (!file.exists(dotR)) 
#   dir.create(dotR)
# M <- file.path(dotR, "Makevars.win")
# if (!file.exists(M)) 
#   file.create(M)
# cat("\nCXX14FLAGS=-O3 -Wno-unused-variable -Wno-unused-function",
#     "CXX14 = $(BINPREF)g++ -m$(WIN) -std=c++1y",
#     "CXX11FLAGS=-O3 -Wno-unused-variable -Wno-unused-function",
#     file = M, sep = "\n", append = TRUE)
# install.packages("rstan")



import pandas as pd
import os
#import climata as cl
#import matplotlib.pyplot as plt
from climata.snotel import RegionDailyDataIO
import datetime  as dt


os.chdir('Z:\SNOTEL\SNOTEL_prediction\PGM')

# pip install climata

#%% Read in one-year prototype

# Test example of load for a basin for one year
data = RegionDailyDataIO(
    start_date="2013-10-01",
    end_date="2014-06-01",
    basin="18010202",
    parameter="SNWD",
)

for series in data:
    for row in series.data:
        print(row)
#%%
# Transform into pandas dataframe
# https://www.earthdatascience.org/tutorials/acquire-and-visualize-usgs-hydrology-data/

datasource = []


for series in data:
    for row in series.data:
        datasource.append(row._asdict())

# unroll the list of lists
# flat_site_names = [item for sublist in site_names for item in sublist]
df = pd.DataFrame(datasource)

#%% One-year prototype EDA
df.head()    
df.tail()    
df.describe(include='all')
pd.crosstab(df['stationtriplet'],columns='count') # Why the hell are tables so complicated?
pd.crosstab(df['begindate'],columns='count') # max begindate = 2003-06-23. Go with Oct 2004 as 1st snow year

#%% Plot the snow over time!
# Works, but then Spyder complains, so it's commented out
#plt.plot(df['date'], df['value'])

#fig, ax = plt.subplots()
#groups = df.groupby('stationtriplet')
#for name, group in groups:
#    #ax.plot(group['date'],group['value'] )
#    ax.plot(group.date,group.value , linestyle='-',  label=name)
#    
#ax.legend(numpoints=1, loc='upper left')
#
#plt.show()

#%% Save the initial prototype dataset
df.to_csv("../DATA/Initial_prototype_data.csv")


#%% Import/transform/save multi-year dataset, with snow year beginning at Oct 1

multiyear_data = RegionDailyDataIO(
    start_date="2004-10-01",
    end_date="2018-09-30",
    basin="18010202",
    parameter="SNWD",
)

multiyear_datasource = []

for series in multiyear_data:
    for row in series.data:
        multiyear_datasource.append(row._asdict())

# flat_site_names = [item for sublist in site_names for item in sublist]
multiyear_df = pd.DataFrame(multiyear_datasource)

multiyear_df.describe(include='all')
multiyear_df.dtypes
multiyear_df['date'] = pd.to_datetime(multiyear_df['date'])

# Add snow year
# Doesn't take leap years, etc into account. But, close enough to a first approximation
# There may be a more sophisticated pandas DateOffset?
snowyear_offset = -273 # months
multiyear_df['snowyeardate'] = multiyear_df['date'] + dt.timedelta(days=snowyear_offset)
multiyear_df['date'] = pd.to_datetime(multiyear_df['date'])

multiyear_df['snowyear'] = pd.DatetimeIndex(multiyear_df['snowyeardate']).year
pd.crosstab(multiyear_df['snowyear'],columns='count') 

# Add Nth day of year
# Subtracts current date from beginning of year, then grabs the dt.days attribute, no +1 offset necessary
multiyear_df['nthdayofyear'] = (multiyear_df['snowyeardate'] - (multiyear_df['snowyeardate'] - pd.offsets.YearBegin())).dt.days  


#multiyear_df['snowyeardate'] - dt.date() + 1

#%% Multi-year plot. Works, but then Spyder complains

#fig, ax = plt.subplots()
#groups = multiyear_df.groupby('stationtriplet')
#for name, group in groups:
#    #ax.plot(group['date'],group['value'] )
#    ax.plot(group.snowyeardate,group.value , linestyle='-',  label=name)
#    
#ax.legend(numpoints=1, loc='upper left')
#
#plt.show()

#%% Save multiyear file

multiyear_df.to_csv("../DATA/Initial_multiyear_prototype_data.csv" )




#%% Test
#pd.Timestamp(year=multiyear_df['snowyear'][0], month= 1, day= 1) 
#
#pd.Timestamp(year=multiyear_df['snowyear'], month= [1] * multiyear_df['snowyear'].size , 
#             day= [1] * multiyear_df['snowyear'].size) 
#[1] * multiyear_df['snowyear'].size

#from pandas.tseries.offsets import *
#(multiyear_df['snowyeardate'] - pd.offsets.YearBegin()) # Get beginning of year!
#
#(multiyear_df['snowyeardate'] - (multiyear_df['snowyeardate'] - pd.offsets.YearBegin())).dt.days 
#stats.describe((multiyear_df['snowyeardate'] - (multiyear_df['snowyeardate'] - pd.offsets.YearBegin())).dt.days ) 
