# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:58:24 2018

@author: Chris
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
import matplotlib.pyplot as plt

os.chdir('Z:\SNOTEL')

# pip install climata

from climata.snotel import RegionDailyDataIO

# Test example of load
data = RegionDailyDataIO(
    start_date="2013-11-01",
    end_date="2014-06-01",
    basin="18010202",
    parameter="SNWD",
)

#%%
# Transform into pandas dataframe
# https://www.earthdatascience.org/tutorials/acquire-and-visualize-usgs-hydrology-data/

datasource = []
stationtriplet = []
flag = []
value = []
date = []


for series in data:
    for row in series.data:
        datasource.append(row[1])
        stationtriplet.append(row[3])
        flag.append(row[11])
        value.append(row[12])
        date.append(row[13])

# unroll the list of lists
# flat_site_names = [item for sublist in site_names for item in sublist]
df = pd.DataFrame({'datasource': datasource, 
                   'stationtriplet': stationtriplet,
                   'flag': flag,
                   'date': date, 
                   'value': value})
df.head()    
df.tail()    
df.describe(include='all')

pd.crosstab(df['stationtriplet'],columns='count') # Why the hell are tables so complicated?

plt.plot(df['date'], df['value'])

fig, ax = plt.subplots()
groups = df.groupby('stationtriplet')
for name, group in groups:
    #ax.plot(group['date'],group['value'] )
    ax.plot(group.date,group.value , linestyle='-',  label=name)
    
ax.legend(numpoints=1, loc='upper left')

plt.show()