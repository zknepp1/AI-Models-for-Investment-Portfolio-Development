import numpy as np
import pandas as pd
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
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('vader_lexicon')

#Data Source
import yfinance as yf
import pynytimes

import time
import re
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



#getting financial data
collection = DataFrameCollection()
collection.retrieve_financial_data()
df_list = collection.return_dataframes()
print(df_list[0].head())


#getting news data
collector = News_Collector()
collector.collect_all_news()
news_data = collector.return_news_data()
print(news_data.head())


print('it worked!')
