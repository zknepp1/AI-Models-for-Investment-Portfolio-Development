

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
financial_data_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/fdata'
news_data_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/ndata'
models_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/models'

check_folder_existence(financial_data_path)
check_folder_existence(news_data_path)
check_folder_existence(models_path)




df_list = []
tickers = ['OKE','MSFT','NVDA','AMD','PAYC','AMZN','GOOGL','^GSPC']


# Trying to read in the financial data
# If the financial data exists locally, it reads in the files (fast way)
# If the financial data does not exist,
#     it retrieves the data and saves it locally
try:
  for filename in os.listdir(financial_data_path):
    if filename.endswith('.csv'):
       file_path = os.path.join(financial_data_path, filename)
       df = pd.read_csv(file_path)
       df_list.append(df)
  print('Try statement worked')

except:
  collection = DataFrameCollection(tickers, '2000-1-1', '2023-1-1')
  collection.retrieve_financial_data()
  collection.save_data()
  df_list = collection.return_dataframes()
  print('Went to except statement')





# Trying to read in the news data
# If the news data exists locally, it reads in the files (fast way)
# If the news data does not exist, it retrieves the data and saves it locally
try:
  news_data_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/ndata/news_data.csv'
  clean_news_data = pd.read_csv(news_data_path)
  print('News data read in try statement!')

except:
  #getting news data
  collector = News_Collector(2015, 2023)
  collector.collect_all_news()
  news_data = collector.return_news_data()

  save_to = '/home/zacharyknepp2012/Knepp_OUDSA5900/ndata/news_data.csv'
  # Cleaning news data
  scrubber = Text_Cleaner(news_data)
  scrubber.scrub_text()
  clean_news_data = scrubber.return_df()
  clean_news_data.to_csv(save_to)
  print('News data went to except statement AHHHHHHHHHHHHHHHHHHHHHHH')




joiner = Join_Data(df_list, clean_news_data)
joiner.pop_market_df()
joiner.combine_dataframes()
dfs_ready = joiner.loop_time_step_creation()



for df in dfs_ready:
  print(df.shape)
  df = df.dropna()
  print(df.shape)




# Tries to read in the financial models.
# If they dont exist, The program will build the models
#      and saved locally for future use
try:
  OKE_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/OKEmodel.h5')
  MSFT_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/MSFTmodel.h5')
  NVDA_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/NVDAmodel.h5')
  AMD_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/AMDmodel.h5')
  PAYC_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/PAYCmodel.h5')
  AMZN_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/AMZNmodel.h5')
  GOOGL_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/GOOGLmodel.h5')
except:
  list_of_models = []
  list_of_mse = []

  count = 0
  for df in dfs_ready:
    builder = Model_Builder(df)
    builder.train_test_scale()
    builder.build_and_optimize_models()
    model = builder.return_best_model()
    mse = builder.return_best_mse()
    model.save('/home/zacharyknepp2012/Knepp_OUDSA5900/models/' + str(tickers[count]) + 'model.h5')
    list_of_models.append(model)
    list_of_mse.append(mse)
    count += 1







print('THE PROGRAM HAS FINISHED EXECUTING! YOU BETTER HAVE A NICE DAY')
print('OR ELSE... >:)')
