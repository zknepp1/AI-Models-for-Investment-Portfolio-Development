import pandas as pd
import re

from dataframe_collector import DataFrameCollection
from news_collecter import News_Collector



class Join_Data:
      def __init__(self, financial_data, news_data):
        self.financial_df = financial_data
        self.news_df = news_data
        self.dates = None
        self.market_df = self.pop_market_df()

        self.combined_dfs = self.combine_dataframes()


        self.dfs_with_timesteps = []

      def return_df(self):
        return self.combined_dfs


      def pop_market_df(self):
        # Remove the last item which is the sp500
        market_copy = self.financial_df.pop()
        #print('hello from the pop market function!!!!!!!!!!!!!!!!!!!')
        #print('now presenting the market copy: :)')
        #print(market_copy)
        date_pattern = r'\d\d\d\d-\d\d-\d\d'
        dates = market_copy['date']

        #print('here are the dates!!!!!!!!!!!!!!!!!!!!!!!!')

        #print(dates)
        new_dates = []

        for i in dates:
          date_found = re.findall(date_pattern, str(i))
          new_dates.append(str(date_found[0]))

        market_open = market_copy['Open']
        clean_dates = new_dates
        market_high = market_copy['High']
        market_low = market_copy['Low']
        market_close = market_copy['Close']
        market_volume = market_copy['Volume']
        market_twenty_roll = market_copy['twenty_day_rolling']
        df = pd.DataFrame({'clean_dates': clean_dates,
                           'market_open': market_open,
                           'market_high': market_high,
                           'market_low': market_low,
                           'market_close': market_close,
                           'market_volume': market_volume,
                           'market_twenty_roll': market_twenty_roll})

        #print('This is what the pop market function returns????')
        #print(df)
        return df



      def combine_dataframes(self):
        # Join based on index
        dfs = []

        for df in self.financial_df:
          date_pattern = r'\d\d\d\d-\d\d-\d\d'
          dates = df['date']
          new_dates = []
          for i in dates:
            date_found = re.findall(date_pattern, str(i))
            new_dates.append(str(date_found[0]))
          df['clean_dates'] = new_dates

        count = 0
        for df in self.financial_df:
          two_df = pd.merge(df, self.market_df, on='clean_dates', how='outer')

          #print('first merge!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
          #print(two_df.head(30))
          #print(two_df.tail(30))


          merged_df = two_df.merge(self.news_df[count], on='clean_dates', how='right')
          #print('second merge!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
          #print(merged_df.head(30))
          #print(merged_df.tail(30))
          count += 1
          dfs.append(merged_df)
        return dfs


      def loop_time_step_creation(self):
        for df in self.combined_dfs:
          ts_df = self.make_time_steps(df)
          self.dfs_with_timesteps.append(ts_df)
        return self.dfs_with_timesteps





#tics = ['AMD', '^GSPC']
#start = '2020-1-1'
#end = '2023-6-1'
#fin = DataFrameCollection(tics, start, end)
#financials = fin.financial_data



# EXAMPLE CODE
#collector = News_Collector(2020, 2022, 9)
#news_data = collector.return_news_data()


#join = Join_Data(financials, news_data)
#df = join.return_df()
#print(df[0])

