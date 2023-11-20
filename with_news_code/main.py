
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)  # Display all columns

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import bigrams, trigrams

#Data Source
import yfinance as yf
import time
import re
import os

#from dataframe_collector import DataFrameCollection
#from news_collecter import News_Collector
#from news_cleanup import Text_Cleaner
from join_data import Join_Data
from model_builder import Model_Builder
#from simulation import Investment_Manager

#function to Check if the folder exists, and if not, create it
def check_folder_existence(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
  else:
    print(f"Folder '{folder_path}' already exists.")


###########
# SECTORS #
###########

# Energy
# Exxon Mobil Corporation: XOM
# Chevron Corporation: CVX
# ConocoPhillips: COP

# Healthcare
# Johnson & Johnson: JNJ
# CVS Health Corporation: CVS
# UnitedHealth Group Incorporated: UNH

# IT
# Intel Corporation - INTC
# 
# 

##########################################################



# Consumer Discretionary
# Netflix, Inc.: NFLX
# Starbucks Corporation: SBUX
# Marriott International, Inc.: MAR

# Consumer Staples
# PepsiCo, Inc.: PEP
# Kellogg Company: K
# The Hershey Company: HSY

# Financials
# Visa Inc.: V
# The Travelers Companies, Inc.: TRV
# CME Group Inc.: CME

# Industrials
# Boeing Company: BA
# 3M Company: MMM
# Delta Air Lines, Inc.: DAL

# Materials
# Sherwin-Williams Company: SHW
# DuPont de Nemours, Inc.: DD
# Mosaic Company: MOS

# Real Estate
# Alexandria Real Estate Equities, Inc.: ARE
# Extra Space Storage Inc.: EXR
# Host Hotels & Resorts, Inc.: HST

# Utilities
# NextEra Energy, Inc.: NEE
# Xcel Energy Inc.: XEL
# PPL Corporation: PPL

# Communication Services
# Comcast Corporation: CMCSA
# Roku, Inc.: ROKU
# Activision Blizzard, Inc.: ATVI



#tics = ['XOM','CVX','COP','JNJ','CVS','UNH',
#        'GOOGL','IBM','AMZN','NFLX','SBUX','MAR',
#        'PEP','K','HSY','V','TRV','CME','BA','MMM','DAL',
#        'SHW','DD','MOS','ARE','EXR','HST',
#        'NEE','XEL','PPL','CMCSA','ROKU','ATVI','^GSPC']


#'CVS' 'UNH'
#tics = ['INTC','XEL','NEE','DD','MOS','BA','MMM','DAL','CME','TRV','V','HSY','K','PEP','SBUX','NFLX','JNJ','XOM','COP','^GSPC']
tics = ['PEP','SBUX','TRV','V','XOM','^GSPC']

tics_no_market = tics[:-1]



# Making all the paths im going to store the data.
# If the folder does not exist, the program will create the folder
data_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/data'
models_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/models'

check_folder_existence(data_path)
check_folder_existence(models_path)




list_train = []
for tic in tics_no_market:
  df = pd.read_csv(f'{data_path}/{tic}_train_df.csv', index_col=0)
  df['ticker'] = tic
  df = df.reset_index()
  list_train.append(df)

print('train data loaded')
print(list_train[0])

list_sim = []
for tic in tics_no_market:
  df = pd.read_csv(f'{data_path}/{tic}_sim_df.csv', index_col=0)
  df['ticker'] = tic
  df = df.reset_index()
  list_sim.append(df)

print('sim data loaded')
print(list_sim[0])




list_of_models = []
list_of_mse = []
count = 0
for df in list_train:
  #print(df)
  #train_df = df.iloc[:-25]
  #sim_df = df.iloc[-25:]

  #train_df.to_csv('/home/zacharyknepp2012/Knepp_OUDSA5900/data/'+ str(tics[count]) + '_train_df.csv', index=False)
  #sim_df.to_csv('/home/zacharyknepp2012/Knepp_OUDSA5900/data/'+ str(tics[count]) +'_sim_df.csv', index=False)

  print('Train df dimensions: ', df.shape)
  print(df.head())


  builder = Model_Builder(df)
  #builder.prep_data()
  #builder.build_rf()

  builder.train_test_scale()
  builder.build_and_optimize_models()

  model = builder.return_best_model()
  mse = builder.return_best_mse()
  print('MSE: ', mse)
  model.save('/home/zacharyknepp2012/Knepp_OUDSA5900/models/' + str(tics[count]) + 'model')
  list_of_models.append(model)
  list_of_mse.append(mse)
  count += 1




#logan = Investment_Manager(check_for_models, loaded_sim_data, loaded_models)
#logan.strategize()

print('THE PROGRAM HAS FINISHED EXECUTING! YOU BETTER HAVE A NICE DAY')
print('OR ELSE... >:)')




