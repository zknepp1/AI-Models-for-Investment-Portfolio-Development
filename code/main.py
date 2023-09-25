import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)  # Display all columns
import matplotlib.pyplot as plt
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

from dataframe_collector import DataFrameCollection
from news_collecter import News_Collector
from news_cleanup import Text_Cleaner
from join_data import Join_Data
from model_builder import Model_Builder


#function to Check if the folder exists, and if not, create it
def check_folder_existence(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
  else:
    print(f"Folder '{folder_path}' already exists.")



# Making all the paths im going to store the data.
# If the folder does not exist, the program will create the folder
data_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/data'
models_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/models'

check_folder_existence(data_path)
check_folder_existence(models_path)

tics = ['AMD', '^GSPC']

try:
  df_list = []
  tiks = tics[:-1]
  for tic in tiks:
    df = pd.read_csv('/home/zacharyknepp2012/Knepp_OUDSA5900/data/'+ str(tics[i]) +'complete_df.csv')
    df_list.append(df)

except:
  start = '2022-1-1'
  end = '2023-10-22'
  fin = DataFrameCollection(tics, start, end)
  financials = fin.financial_data

  collector = News_Collector(2022, 2024, 10)
  news_data = collector.return_news_data()

  join = Join_Data(financials, news_data)
  df_list = join.return_df()
  # SAVING DATA  FOR FUTURE USE
  i = 0
  for df in df_list:
    df.to_csv('/home/zacharyknepp2012/Knepp_OUDSA5900/data/'+ str(tics[i]) +' _complete_df.csv', index=False)
    i += 1






d = df_list[0]
train_df = d.iloc[:-25]
sim_df = d.iloc[-25:]


builder = Model_Builder(df)
builder.train_test_scale()
builder.build_and_optimize_models()





# Tries to read in the financial models.
# If they dont exist, The program will build the models
#      and saved locally for future use
#try:
#  OKE_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/OKEmodel.h5')
#  MSFT_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/MSFTmodel.h5')
#  NVDA_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/NVDAmodel.h5')
#  AMD_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/AMDmodel.h5')
#  PAYC_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/PAYCmodel.h5')
#  AMZN_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/AMZNmodel.h5')
#  GOOGL_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/GOOGLmodel.h5')
#except:
#  list_of_models = []
#  list_of_mse = []

#  count = 0
#  for df in dfs_ready:
#    builder = Model_Builder(df)
#    builder.train_test_scale()
#    builder.build_and_optimize_models()
#    model = builder.return_best_model()
#    mse = builder.return_best_mse()
#    model.save('/home/zacharyknepp2012/Knepp_OUDSA5900/models/' + str(tickers[count]) + 'model.h5')
#    list_of_models.append(model)
#    list_of_mse.append(mse)
#    count += 1







print('THE PROGRAM HAS FINISHED EXECUTING! YOU BETTER HAVE A NICE DAY')
print('OR ELSE... >:)')
