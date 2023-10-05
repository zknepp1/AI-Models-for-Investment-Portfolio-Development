import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from dataframe_collector import DataFrameCollection
from news_collecter import News_Collector
from news_cleanup import Text_Cleaner
from join_data import Join_Data

class Investment_Manager:
    def __init__(self, tickers, data, list_of_models):
        self.tickers_list = tickers
        self.data_list = data
        self.models_list = list_of_models
        self.vars = ['Open', 'High', 'Low','Close','Volume',
                  'five_day_rolling','ten_day_rolling','twenty_day_rolling',
                  'cleaned_pos','cleaned_neg','recession', 'fomc','inflation',
                  'cpi','unemployment','gdp','bubble','bear','bearish','bull',
                  'bullish','acquires','acquisition','merger','war','vix','volatility',
                  'market_open', 'market_high','market_low', 'market_close',
                  'market_volume','market_twenty_roll']

        self.preds = self.make_predictions()
        self.prepped_data = self.prep_data()

        self.strategy()


    def make_predictions(self):
        print(len(self.data_list))
        count = 0
        preds_list = []
        for df in self.data_list:
          #print(df)
          #df = df.dropna()
          #print(df)
          print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
          X = df[self.vars]
          #print(X)

          X = X.dropna()
          #print(X)
          print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
          # Convert to numpy arrays
          X_array = np.array(X)

          scaler = StandardScaler()
          scaler.fit(X_array)
          X_scaled = scaler.transform(X_array)
          #print(X_scaled)
          m = 33
          timesteps = 1
          X_reshaped = X_scaled.reshape(X_scaled.shape[0], timesteps, m)
          preds = self.models_list[count].predict(X_reshaped)
          preds_list.append(preds)
          count += 1
        return preds_list


    def prep_data(self):
        count = 0
        df_list = []
        for i in self.data_list:
          df = pd.DataFrame()
          df['dates'] = i['clean_dates']
          df['close'] = i['Close']
          df = df.dropna()
          df = df.reset_index(drop=True)
          p = self.preds[count]
          p = p.reshape(-1, 1)
          pre = pd.DataFrame(p)
          df['preds'] = pre
          df_list.append(df)
          count += 1
        return df_list


    def strategy(self):
        print(self.prepped_data)


#tickers = ['AMD', '^GSPC']
#l = Investment_Manager(tickers, '2005-1-1')
#l.make_predictions()

#print('THIS IS THE END...')

