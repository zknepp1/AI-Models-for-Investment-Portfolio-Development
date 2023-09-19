import pandas as pd
import re

class Join_Data:
      def __init__(self, financial_data, news_data):
        self.financial_df = financial_data
        self.news_df = news_data
        self.market_df = pd.DataFrame()

        self.combined_dfs = []
        self.dfs_with_timesteps = []



      def view_df(self, num):
        if num ==  1:
          print(self.financial_df[0].head())
        elif num == 2:
          print(self.news_df.head())
        elif  num == 3:
          print(self.market_df.head())



      def pop_market_df(self):
        # Remove the last item which is the sp500
        market_copy = self.financial_df.pop()
        df = pd.DataFrame()

        date_pattern = r'\d\d\d\d-\d\d-\d\d'
        dates = market_copy['date']

        new_dates = []
        for i in dates:
          date_found = re.findall(date_pattern, str(i))
          new_dates.append(str(date_found[0]))

        df['clean_dates'] = new_dates
        df['market_open'] = market_copy['Open']
        df['market_high'] = market_copy['High']
        df['market_low'] = market_copy['Low']
        df['market_close'] = market_copy['Close']
        df['market_volume'] = market_copy['Volume']
        df['market_twenty_roll'] = market_copy['twenty_day_rolling']
        self.market_df = df


      def column_names(self, i):
        new_column_names = {'Open': 'Open' + str(i),
                            'High': 'High' + str(i),
                            'Low': 'Low' + str(i),
                            'Close': 'Close' + str(i),
                            'Volume': 'Volume' + str(i),
                            'Five_day_rolling': 'Five_day_rolling' + str(i),
                            'Ten_day_rolling': 'Ten_day_rolling' + str(i),
                            'Twenty_day_rolling': 'Twenty_day_rolling' + str(i),
                            'cleaned_pos': 'cleaned_pos' + str(i),
                            'cleaned_neg': 'cleaned_neg' + str(i),
                            'recession': 'recession' + str(i),
                            'fomc': 'fomc' + str(i),
                            'inflation': 'inflation' + str(i),
                            'cpi': 'cpi' + str(i),
                            'unemployment': 'unemployment' + str(i),
                            'gdp': 'gdp' + str(i),
                            'bubble': 'bubble' + str(i),
                            'bear': 'bear' + str(i),
                            'bearish': 'bearish' + str(i),
                            'bull': 'bull' + str(i),
                            'bullish': 'bullish' + str(i),
                            'acquires': 'acquires' + str(i),
                            'acquisition': 'acquisition' + str(i),
                            'merger': 'merger' + str(i),
                            'war': 'war' + str(i),
                            'vix': 'vix' + str(i),
                            'volatility': 'volatility' + str(i),
                            'market_open': 'market_open' + str(i),
                            'market_high': 'market_high' + str(i),
                            'market_low': 'market_low' + str(i),
                            'market_close': 'market_close' + str(i),
                            'market_volume': 'market_volume' + str(i),
                            'market_five_roll': 'market_five_roll' + str(i),
                            'market_ten_roll': 'market_ten_roll' + str(i),
                            'market_twenty_roll': 'market_twenty_roll' + str(i)}
        return new_column_names


      def make_time_steps(self, df):
        # Shift the variables down by 1 row
        shifted_df_1 = df.shift(1)
        shifted_df_2 = df.shift(2)
        shifted_df_3 = df.shift(3)
        shifted_df_4 = df.shift(4)
        shifted_df_5 = df.shift(5)


        X = ['Open', 'High', 'Low',
             'Close','Volume','five_day_rolling','ten_day_rolling','twenty_day_rolling',
             'cleaned_pos','cleaned_neg',
             'recession', 'fomc','inflation','cpi','unemployment','gdp','bubble',
             'bear','bearish','bull','bullish','acquires','acquisition',
             'merger','war','vix','volatility',
             'market_open', 'market_high',
             'market_low', 'market_close', 'market_volume',
             'market_twenty_roll']

        X_with_target = ['Target','Open', 'High', 'Low',
             'Close','Volume','five_day_rolling','ten_day_rolling','twenty_day_rolling',
             'cleaned_pos','cleaned_neg',
             'recession', 'fomc','inflation','cpi','unemployment','gdp','bubble',
             'bear','bearish','bull','bullish','acquires','acquisition',
             'merger','war','vix','volatility',
             'market_open', 'market_high', 'market_low', 'market_close',
             'market_volume', 'market_twenty_roll']

        df = df[X_with_target]

        shifted_df_1 = shifted_df_1[X]
        new_column_names_1 = self.column_names(1)
        shifted_df_1.rename(columns=new_column_names_1, inplace=True)

        shifted_df_2 = shifted_df_2[X]
        new_column_names_2 = self.column_names(2)
        shifted_df_2.rename(columns=new_column_names_2, inplace=True)

        shifted_df_3 = shifted_df_3[X]
        new_column_names_3 = self.column_names(3)
        shifted_df_3.rename(columns=new_column_names_3, inplace=True)

        shifted_df_4 = shifted_df_4[X]
        new_column_names_4 = self.column_names(4)
        shifted_df_4.rename(columns=new_column_names_4, inplace=True)

        shifted_df_5 = shifted_df_5[X]
        new_column_names_5 = self.column_names(5)
        shifted_df_5.rename(columns=new_column_names_5, inplace=True)

        # Append the shifted variables to the original DataFrame
        result_df = pd.concat([df, shifted_df_1, shifted_df_2, shifted_df_3, shifted_df_4, shifted_df_5], axis=1)
        return result_df


      def combine_dataframes(self):
        # Join based on index
        for df in self.financial_df:
          date_pattern = r'\d\d\d\d-\d\d-\d\d'
          dates = df['date']
          new_dates = []
          for i in dates:
            date_found = re.findall(date_pattern, str(i))
            new_dates.append(str(date_found[0]))
          df['clean_dates'] = new_dates

        for df in self.financial_df:
          two_df = pd.merge(df, self.market_df, on='clean_dates', how='outer')
          merged_df = pd.merge(two_df, self.news_df, on='clean_dates', how='outer')
          merged_df = merged_df.dropna()
          self.combined_dfs.append(merged_df)

        print()
        print('join_data.py')
        print(self.combined_dfs[0])

        return self.combined_dfs


      def loop_time_step_creation(self):
        for df in self.combined_dfs:
          ts_df = self.make_time_steps(df)
          self.dfs_with_timesteps.append(ts_df)
        return self.dfs_with_timesteps



