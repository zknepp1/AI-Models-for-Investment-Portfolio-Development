import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)  # Display all columns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Data Source
import yfinance as yf
import pynytimes
import time
import re
import os


#import tensorflow as tf
#from keras.models import Sequential
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
#from keras.layers import Embedding,Dense,LSTM,Dropout,Flatten,BatchNormalization,Conv1D,GlobalMaxPooli>
#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.callbacks import EarlyStopping
#tf.random.set_seed(7)

from dataframe_collector import DataFrameCollection
from news_collecter import News_Collector
from news_cleanup import Text_Cleaner
from join_data import Join_Data

#getting financial data
financial_data_path = '/home/zacharyknepp2012/Knepp_OUDSA5900/fdata'
df_list = []




try:
  for filename in os.listdir(financial_data_path):
    if filename.endswith('.csv'):
       file_path = os.path.join(financial_data_path, filename)
       df = pd.read_csv(file_path)
       df_list.append(df)
  print('Try statement worked')

except:
  collection = DataFrameCollection()
  collection.retrieve_financial_data()
  collection.save_data()
  df_list = collection.return_dataframes()
  print('Went to except statement')

#print(df_list[0].head())





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





#print(clean_news_data.head())
#print(clean_news_data.shape)





joiner = Join_Data(df_list, clean_news_data)
joiner.pop_market_df()
joiner.combine_dataframes()
#dfs_ready = joiner.loop_time_step_creation()
#print(dfs_ready[0].head(30))

#print(dfs_ready[0]['cleaned_pos'])




print('THE PROGRAM HAS FINISHED EXECUTING! YOU BETTER HAVE A NICE DAY')
print('OR ELSE... >:)')
