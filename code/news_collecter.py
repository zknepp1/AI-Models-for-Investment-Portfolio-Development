import pandas as pd
import pynytimes
import time
import requests
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import statistics


#from ast import Match
from datetime import datetime



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('vader_lexicon')

pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all columns


#This class is meant to collect news articles from nyt between the years 2000-2023
# The packages used for this class is pynytimes
class News_Collector:
    def __init__(self, tics):

        self.tickers = tics


        self.from_ = ['20220101T0000','20220201T0000','20220301T0000','20220401T0000',
                      '20220501T0000','20220601T0000','20220701T0000','20220801T0000',
                      '20220901T0000','20221001T0000','20221101T0000','20221201T0000',
                      '20230101T0000','20230201T0000','20230301T0000','20230401T0000',
                      '20230501T0000','20230601T0000','20230701T0000','20230801T0000',
                      '20230901T0000','20231001T0000']

        self.to_ = ['20220201T0000','20220301T0000','20220401T0000','20220501T0000',
                    '20220601T0000','20220701T0000','20220801T0000','20220901T0000',
                    '20221001T0000','20221101T0000','20221201T0000','20230101T0000',
                    '20230201T0000','20230301T0000','20230401T0000','20230501T0000',
                    '20230601T0000','20230701T0000','20230801T0000','20230901T0000',
                    '20231001T0000','20231101T0000']

        self.news_data = self.collect_all_news()
        self.data_with_topics = self.loop_through_count_topics()
        self.mode_data = self.loop_through_mode_of_labels()


        #self.clean_news_data = self.clean_news_data()
        #self.clusters = self.cluster_news()
        #self.sentiment = self.calculate_sentiment()
        #self.count_data = self.look_for_words()


    # Collects all news data since the year 2000
    def collect_all_news(self):
        base_url = 'https://www.alphavantage.co/query?'
        function = 'NEWS_SENTIMENT'
        api_key = '2BVQAFUBRB5U2BZR'
        limit='1000'
        tickers = self.tickers

        list_of_dfs = []
        for ticker in tickers:
          dfs = []
          for i in range(len(self.from_)):
            pattern = r'(\d{4})(\d{2})(\d{2})'
            time_from= self.from_[i]
            time_to = self.to_[i]
            try:
              response = requests.get(f'{base_url}function={function}&tickers={ticker}&time_from={time_from}&time_to={time_to}&limit={limit}&apikey={api_key}')
              data = response.json()
              df = pd.DataFrame(data['feed'])
              dates = [str(date) for date in df['time_published']]
              m = []
              for date in dates:
                match = re.search(pattern, date)
                if match:
                  formatted_date = '-'.join(match.groups())
                  m.append(formatted_date)
                else:
                  print("No date found in the input string.")

              df['clean_dates'] = m
              df['ticker'] = ticker
              dfs.append(df)
              time.sleep(3)

            except:
              print('Data for this month does not exist')

          list_of_dfs.append(dfs)

        combined_list = []
        for df in list_of_dfs:
          combined_df = pd.DataFrame()
          for i in df:
            combined_df = pd.concat([combined_df, i])
          combined_list.append(combined_df)


        agg_dict = {
            'title': list,
            'summary': list,
            'topics': list,
            'overall_sentiment_score': list,
            'overall_sentiment_label': list,
            'ticker_sentiment': list
        }

        grouped_df_list = []
        for df in combined_list:
          grouped_df = df.groupby('clean_dates').agg(agg_dict).reset_index()
          grouped_df_list.append(grouped_df)

        return grouped_df_list


    def loop_through_count_topics(self):
        dfs_with_topics = []
        for df in self.news_data:
          tdf = self.count_topics(df)
          dfs_with_topics.append(tdf)
        return dfs_with_topics


    def count_topics(self, df):
        topix = df['topics']
        print(len(topix))
        copy = df.copy()

        tech_list = []
        block_list = []
        economy_list = []
        ipo_list = []
        retail_list = []
        finmarket_list = []
        manu_list = []
        real_est_list = []
        fin_list = []
        ls_list = []
        earnings_list = []
        merge_list = []
        energy_list = []
        ef_list = []
        em_list = []
        for top in topix:
          for t in top:
            tech = 0
            block = 0
            economy = 0
            ipo = 0
            retail = 0
            finmarket = 0
            manu = 0
            real_est = 0
            fin = 0
            ls = 0
            earnings = 0
            merge = 0
            energy = 0
            ef = 0
            em = 0

            for i in t:
              if i['topic'] == 'Technology':
                tech += 1
              elif i['topic'] == 'Blockchain':
                block += 1
              elif i['topic'] == 'Economy - Monetary':
                economy += 1
              elif i['topic'] == 'IPO':
                ipo += 1
              elif i['topic'] == 'Retail & Wholesale':
                retail += 1
              elif i['topic'] == 'Financial Markets':
                finmarket += 1
              elif i['topic'] == 'Manufacturing':
                manu += 1
              elif i['topic'] == 'Real Estate & Construction':
                real_est += 1
              elif i['topic'] == 'Finance':
                fin += 1
              elif i['topic'] == 'Life Sciences':
                ls += 1
              elif i['topic'] == 'Earnings':
                earnings += 1
              elif i['topic'] == 'Mergers & Acquisitions':
                merge += 1
              elif i['topic'] == 'Energy & Transportation':
                energy += 1
              elif i['topic'] == 'Economy - Fiscal':
                ef += 1
              elif i['topic'] == 'Economy - Macro':
                em += 1

          tech_list.append(tech)
          block_list.append(block)
          economy_list.append(economy)
          ipo_list.append(ipo)
          retail_list.append(retail)
          finmarket_list.append(finmarket)
          manu_list.append(manu)
          real_est_list.append(real_est)
          fin_list.append(fin)
          ls_list.append(ls)
          earnings_list.append(earnings)
          merge_list.append(merge)
          energy_list.append(energy)
          ef_list.append(ef)
          em_list.append(em)
        copy['Technology'] = tech_list
        copy['Blockchain'] = block_list
        copy['Economy_Monetary'] = economy_list
        copy['IPO'] = ipo_list
        copy['Retail_Wholesale'] = retail_list
        copy['Financial_Markets'] = finmarket_list
        copy['Manufacturing'] = manu_list
        copy['Real_Estate'] = real_est_list
        copy['Finance'] = fin_list
        copy['Life_Sciences'] = ls_list
        copy['Earnings'] = earnings_list
        copy['Mergers'] = merge_list
        copy['Energy'] = energy_list
        copy['Economy_Fiscal'] = ef_list
        copy['Economy_Macro'] = em_list

        return copy



    def loop_through_mode_of_labels(self):
        dfs_list = []
        count = 0
        for df in self.data_with_topics:
          df_modes = self.mode_of_labels(df)
          df_modes['ticker'] = self.tickers[count]
          dfs_list.append(df_modes)
          count += 1

        return dfs_list



    def mode_of_labels(self, df):
        copy = df.copy()
        sentlabels = df['overall_sentiment_label']
        labels = []
        for i in sentlabels:
          labels.append(statistics.mode(i))

        copy['sentiment_labels'] = labels
        # Convert 'Category' column into binary variables
        copy = pd.get_dummies(copy, columns=['sentiment_labels'], prefix=['sentiment_labels'])
        copy['sentiment_labels_Bullish'] = copy['sentiment_labels_Bullish'].astype(int)
        copy['sentiment_labels_Neutral'] = copy['sentiment_labels_Neutral'].astype(int)
        copy['sentiment_labels_Somewhat-Bearish'] = copy['sentiment_labels_Somewhat-Bearish'].astype(int)
        copy['sentiment_labels_Somewhat-Bullish'] = copy['sentiment_labels_Somewhat-Bullish'].astype(int)

        return copy


    # Saves the dataframe locally
    def save_all_news_data(self):
        df = pd.DataFrame({'dt': self.dt, 'headline': self.headline, 'snippet': self.snippet})
        df.to_csv('/content/drive/My Drive/news/newsdata.csv')
        print('Saved data to newsdata.csv')


    def return_news_data(self):
        return self.mode_data







# EXAMPLE CODE
#collector = News_Collector()
#news_data = collector.return_news_data()
#print(news_data)






