import time
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)  # Display all columns
#import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import bigrams, trigrams

#Data Source
import yfinance as yf
import pynytimes
import time
import re
import os
import glob

import warnings
warnings.filterwarnings("ignore")



from dataframe_collector import DataFrameCollection
#from news_collecter import News_Collector
#from news_cleanup import Text_Cleaner
from join_data import Join_Data
from model_builder import Model_Builder
from simulation import Investment_Manager

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
# Activision Blizzard



print('###############################################################')
print('###############################################################')

print()
print()
print('Hello User')
print()
print('This program takes stock labels and will make recomendations on which stocks and how many of which stock to buy')
print('Keep in mind that each model takes about 20 minutes to build.')


user_inputs = []

while True:
    user_input = input("Please enter a stock label (or type 'exit' to finish): ")
    if user_input.lower() == 'exit':
        break
    user_inputs.append(user_input)

print("You entered:", user_inputs)



tics = user_inputs
tics.append('^GSPC')



labels = tics[:-1]


# Start timing the program
start_time = time.time()




# Making all the paths im going to store the data.
# If the folder does not exist, the program will create the folder
data_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/data'
models_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/models'

check_folder_existence(data_path)
check_folder_existence(models_path)




#train_file_pattern = '*_train_df.csv'
#sim_file_pattern = '*_sim_df.csv'
#train_csv_files = glob.glob(os.path.join(data_path, train_file_pattern))
#sim_csv_files = glob.glob(os.path.join(data_path, sim_file_pattern))





current_date = datetime.now().strftime("%Y-%m-%d")
print(current_date)

# Pulling financial data between these dates
start = '2022-1-1'
end = current_date
fin = DataFrameCollection(tics, start, end)
financials = fin.financial_data







# Joining the ticker data with market data and news data
join = Join_Data(financials)
df_list = join.return_df()


#print(df_list[0])


# Model building and data/model storage


list_of_models = []
list_of_mse = []
count = 0
for df in df_list:
  # Building model

  # train df used to train the models
  # sim df used for simulation and future prediction
  train_df = df.iloc[:-25]
  sim_df = df.iloc[-25:]

  train_df.to_csv('/home/zacharyknepp2012/Knepp_OUDSA5900/data/'+ str(tics[count]) + '_train_df.csv', index=False)
  sim_df.to_csv('/home/zacharyknepp2012/Knepp_OUDSA5900/data/'+ str(tics[count]) +'_sim_df.csv', index=False)

  print('Train df dimensions: ', train_df.shape)
  print(train_df.head())


  builder = Model_Builder(train_df)
  builder.train_test_scale()
  builder.build_and_optimize_models()

  model = builder.return_best_model()
  mse = builder.return_best_mse()
  print('MSE: ', mse)
  model.save('/home/zacharyknepp2012/Knepp_OUDSA5900/models/' + str(labels[count]) + 'model')
  list_of_models.append(model)
  list_of_mse.append(mse)
  count += 1

# end timing the program
end_time = time.time()




l = Investment_Manager(labels)
l.maximize_returns()

execution_time = end_time - start_time  # Calculate the execution time
print("Execution time: {:.2f} seconds".format(execution_time))

print('THE PROGRAM HAS FINISHED EXECUTING! HAVE A NICE DAY')


