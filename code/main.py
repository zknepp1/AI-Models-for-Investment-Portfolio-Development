
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
from simulation import Investment_Manager

#function to Check if the folder exists, and if not, create it
def check_folder_existence(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
  else:
    print(f"Folder '{folder_path}' already exists.")


tics = ['AMD','GOOGL', '^GSPC']
check_for_models = tics[:-1]



# Making all the paths im going to store the data.
# If the folder does not exist, the program will create the folder
data_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/data'
models_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/models'

check_folder_existence(data_path)
check_folder_existence(models_path)





try:
  loaded_models = []
  loaded_sim_data = []
  for tic in check_for_models:
    model = tf.keras.models.load_model(models_path + '/' + str(tic) + 'model')
    loaded_models.append(model)

    sim_df = pd.read_csv(data_path + '/' + str(tic) + '_sim_df.csv')
    loaded_sim_data.append(sim_df)


    print('The try statement was successful!!!!!!!!!!!!!!!!!!!!!!')



except:
  start = '2015-1-1'
  end = '2023-10-22'
  fin = DataFrameCollection(tics, start, end)
  financials = fin.financial_data
  collector = News_Collector(2015, 2024, 10)
  news_data = collector.return_news_data()
  join = Join_Data(financials, news_data)
  df_list = join.return_df()

  list_of_models = []
  list_of_mse = []
  count = 0
  for df in df_list:
    #print(df)
    train_df = df.iloc[:-25]
    sim_df = df.iloc[-25:]

    train_df.to_csv('/home/zacharyknepp2012/Knepp_OUDSA5900/data/'+ str(tics[count]) + '_train_df.csv', index=False)
    sim_df.to_csv('/home/zacharyknepp2012/Knepp_OUDSA5900/data/'+ str(tics[count]) +'_sim_df.csv', index=False)

    builder = Model_Builder(train_df)
    builder.train_test_scale()
    builder.build_and_optimize_models()
    model = builder.return_best_model()
    mse = builder.return_best_mse()
    model.save('/home/zacharyknepp2012/Knepp_OUDSA5900/models/' + str(tics[count]) + 'model')
    list_of_models.append(model)
    list_of_mse.append(mse)
    count += 1




logan = Investment_Manager(check_for_models, loaded_sim_data, loaded_models)


print('THE PROGRAM HAS FINISHED EXECUTING! YOU BETTER HAVE A NICE DAY')
print('OR ELSE... >:)')




