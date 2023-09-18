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




#print(clean_news_data)
text = clean_news_data['cleaned_text']


word_list = [word for word in text]

text_by_day = []
for i in word_list:
  combined_string = ''.join(i)
  words = re.findall(r'\w+', combined_string)
  text_by_day.append(words)




recession_list = []
fomc_list = []
inflation_list = []
cpi_list = []
unemployment_list = []
gdp_list = []
bubble_list = []
bear_list = []
bearish_list = []
bull_list = []
bullish_list = []
acquires_list = []
acquisition_list = []
merger_list = []
war_list = []
vix_list = []
volatility_list = []

rate_cuts_list = []
rate_hikes_list = []
beat_earnings_list = []
beat_eps_list = []
beat_revenue_list = []
missed_earnings_list = []
missed_eps_list = []
missed_revenue_list = []
dividend_cut_list = []
dividend_raise_list = []

for day in text_by_day:
  recession = 0
  fomc = 0
  inflation = 0
  cpi = 0
  unemployment = 0
  gdp = 0
  bubble = 0
  bear = 0
  bearish = 0
  bull = 0
  bullish = 0
  acquires = 0
  acquisition = 0
  merger = 0
  war = 0
  vix = 0
  volatility = 0

  rate_cuts = 0
  rate_hikes = 0
  beat_earnings = 0
  beat_eps = 0
  beat_revenue = 0
  missed_earnings = 0
  missed_eps = 0
  missed_revenue = 0
  dividend_cut = 0
  dividend_raise = 0
  # Generate bigrams
  bi_grams = list(bigrams(day))
  for i in bi_grams:
    i = i[0].lower() + ' ' + i[1].lower()
    if i.lower() == 'rate cuts' or i.lower() == 'rate cut':
      rate_cuts += 1
    elif i.lower() == 'rate hikes' or i.lower() == 'rate hike':
      rate_hikes += 1
    elif i.lower() == 'beat earnings':
      beat_earnings += 1
    elif i.lower() == 'beat eps':
      beat_eps += 1
    elif i.lower() == 'beat revenue':
      beat_revenue += 1
    elif i.lower() == 'missed earnings':
      missed_earnings += 1
    elif i.lower() == 'missed eps':
      missed_eps += 1
    elif i.lower() == 'missed revenue':
      missed_revenue += 1
    elif i.lower() == 'dividend cuts' or i.lower() == 'dividend cut':
      dividend_cut += 1
    elif i.lower() == 'dividend raise' or i.lower() == 'dividend raises':
      dividend_raise += 1

  rate_cuts_list.append(rate_cuts)
  rate_hikes_list.append(rate_hikes)
  beat_earnings_list.append(beat_earnings)
  beat_eps_list.append(beat_eps)
  beat_revenue_list.append(beat_revenue)
  missed_earnings_list.append(missed_earnings)
  missed_eps_list.append(missed_eps)
  missed_revenue_list.append(missed_revenue)
  dividend_cut_list.append(dividend_cut)
  dividend_raise_list.append(dividend_raise)



  for i in day:
    if i.lower() == 'recession':
      recession += 1
    elif i.lower() == 'fomc':
      fomc += 1
    elif i.lower() == 'inflation':
      inflation += 1
    elif i.lower() == 'cpi':
      cpi += 1
    elif i.lower() == 'unemployment':
      unemployment += 1
    elif i.lower() == 'gdp':
      gdp += 1
    elif i.lower() == 'bubble':
      bubble += 1
    elif i.lower() == 'bear':
      bear += 1
    elif i.lower() == 'bearish':
      bearish += 1
    elif i.lower() == 'bull':
      bull += 1
    elif i.lower() == 'bullish':
      bullish += 1
    elif i.lower() == 'acquires':
      acquires += 1
    elif i.lower() == 'acquisition':
      acquisition += 1
    elif i.lower() == 'merger':
      merger += 1
    elif i.lower() == 'war':
      war += 1
    elif i.lower() == 'vix':
      vix += 1
    elif i.lower() == 'volatility':
      volatility += 1

  recession_list.append(recession)
  fomc_list.append(fomc)
  inflation_list.append(inflation)
  cpi_list.append(cpi)
  unemployment_list.append(unemployment)
  gdp_list.append(gdp)
  bubble_list.append(bubble)
  bear_list.append(bear)
  bearish_list.append(bearish)
  bull_list.append(bull)
  bullish_list.append(bullish)
  acquires_list.append(acquires)
  acquisition_list.append(acquisition)
  merger_list.append(merger)
  war_list.append(war)
  vix_list.append(vix)
  volatility_list.append(volatility)




print(rate_cuts_list)





#joiner = Join_Data(df_list, clean_news_data)
#joiner.pop_market_df()
#joiner.combine_dataframes()
#dfs_ready = joiner.loop_time_step_creation()


#for df in dfs_ready:
#  print(df.shape)
#  df = df.dropna()
#  print(df.shape)



#builder = Model_Builder(dfs_ready)
#builder.train_test_scale()
#builder.build_and_optimize_models()
#builder.plot_best_model_results()






print('THE PROGRAM HAS FINISHED EXECUTING! YOU BETTER HAVE A NICE DAY')
print('OR ELSE... >:)')



