import pulp
import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#from dataframe_collector import DataFrameCollection
#from news_collecter import News_Collector
#from news_cleanup import Text_Cleaner
#from join_data import Join_Data

class Investment_Manager:
    def __init__(self, tickers):
        self.tickers_list = tickers
        self.data_list = self.loop_load_sim_data()
        self.models_list = self.loop_load_models()




        self.vars = ['Open', 'High', 'Low','Close','Volume',
                  'five_day_rolling','ten_day_rolling','twenty_day_rolling',
                  'Technology','Blockchain','Economy_Monetary','IPO',
                  'Retail_Wholesale','Financial_Markets','Manufacturing',
                  'Real_Estate','Finance','Life_Sciences','Earnings',
                  'Mergers','Energy','Economy_Fiscal','Economy_Macro',
                  'sentiment_labels_Bullish','sentiment_labels_Bearish',
                  'sentiment_labels_Neutral','sentiment_labels_Somewhat-Bullish',
                  'sentiment_labels_Somewhat-Bearish','average_sentiment',
                  'market_open', 'market_high','market_low', 'market_close',
                  'market_volume','market_twenty_roll']


        self.preds = self.make_predictions()
        self.data_with_preds = self.data_with_preds()

        self.final_df = self.prep_for_opt()


        #self.strategy()


    def loop_load_sim_data(self):
        print('loop sim start')
        sim_list = []
        for i in self.tickers_list:
          sim_list.append(self.load_sim_data(i))
        print('loop sim end')
        return sim_list


    def load_sim_data(self, tic):
        path = '/home/zacharyknepp2012/Knepp_OUDSA5900/data/'+ str(tic) +'_sim_df.csv'
        df = pd.read_csv(path)
        return df


    def loop_load_models(self):
        print('loop model start')
        model_list = []
        for i in self.tickers_list:
          model_list.append(self.load_model(i))
        print('loop model end')
        return model_list


    def load_model(self, tic):
        path = '/home/zacharyknepp2012/Knepp_OUDSA5900/models/' + str(tic) + 'model'
        loaded_model = tf.keras.models.load_model(path)
        return loaded_model


    def make_predictions(self):
        print(len(self.data_list))
        count = 0
        preds_list = []
        for df in self.data_list:
          print(df.shape)
          print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
          X = df[self.vars]

          X = X.dropna()
          print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
          # Convert to numpy arrays
          X_array = np.array(X)

          scaler = StandardScaler()
          scaler.fit(X_array)
          X_scaled = scaler.transform(X_array)
          #print(X_scaled)
          m = 35
          timesteps = 1
          X_reshaped = X_scaled.reshape(X_scaled.shape[0], timesteps, m)
          preds = self.models_list[count].predict(X_reshaped)
          preds_list.append(preds)
          count += 1
        return preds_list


    def data_with_preds(self):
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



    def calculate_return(self, df):
        # Calculate the return as a percentage
        df['return'] = ((df['preds'] - df['close']) / df['close']) * 100
        return df


    def calculate_sharpe_ratio(self, df):
        risk_free_rate = 0.001
        stddev = np.std(df['return'], ddof=1)
        mean_return = df['return'].mean()
        sharpe_ratio = (mean_return - risk_free_rate) / stddev
        return sharpe_ratio, mean_return, stddev


    def prep_for_opt(self):
        std_per_df = []
        return_per_df = []
        sharpe_per_df = []
        open = []
        count = 0
        for df in self.data_with_preds:
          df = self.calculate_return(df)
          sharpe_ratio, mean_return, std_dev = self.calculate_sharpe_ratio(df)
          sharpe_per_df.append(sharpe_ratio)
          return_per_df.append(mean_return)
          std_per_df.append(std_dev)
          og_df = self.data_list[count]
          op = og_df['Open']

          print(op.iloc[24])
          open.append(op.iloc[24])
          count += 1

        df = pd.DataFrame({'ticker':self.tickers_list, 'sharpe':sharpe_per_df, 'open':open, 'return':return_per_df, 'std':std_per_df})
        print(df)
        print(df.shape)
        return df


    def maximize_returns(self):
        # Create a PuLP LP problem to optimize the number of stocks to buy
        lp_problem = pulp.LpProblem("Optimize_Stocks_To_Buy", pulp.LpMaximize)

        # Define decision variables for the number of shares to buy for each stock
        shares_to_buy = pulp.LpVariable.dicts("Shares_To_Buy", self.final_df['ticker'], lowBound=0, cat='Integer')

        # Define the objective function (maximize expected return)
        lp_problem += pulp.lpSum([self.final_df.at[i, 'return'] * shares_to_buy[self.final_df.at[i, 'ticker']] for i in self.final_df.index])

        # Add a constraint to limit the total budget spent
        total_budget = 5000
        lp_problem += pulp.lpSum([self.final_df.at[i, 'open'] * shares_to_buy[self.final_df.at[i, 'ticker']] for i in self.final_df.index]) <= total_budget

        # Add a constraint to ensure a minimum required Sharpe ratio
        risk_free_rate = 0.001
        min_sharpe_ratio = 0.1
        max_concentration = 0.05
        lp_problem += pulp.lpSum([(self.final_df.at[i, 'return'] - risk_free_rate) * shares_to_buy[self.final_df.at[i, 'ticker']] / self.final_df.at[i, 'std'] for i in self.final_df.index]) >= min_sharpe_ratio

        #lp_problem +=pulp.lpSum([( shares_to_buy[stock] <= max_concentration * shares_to_buy[stock]


        # Solve the linear programming problem
        lp_problem.solve()


        # Print the number of shares to buy for each stock
        for stock in self.final_df['ticker']:
          num_shares = shares_to_buy[stock].varValue
          if num_shares > 0:
            print(f"Buy {num_shares} shares of {stock}")





print('hello world')

tickers = ['INTC','XEL','NEE','DD','MOS','BA','MMM','DAL','CME','TRV','V','HSY','K','PEP','SBUX','NFLX','JNJ','XOM','COP']
l = Investment_Manager(tickers)
l.maximize_returns()

#print('THIS IS THE END...')



#sim_df.to_csv('/home/zacharyknepp2012/Knepp_OUDSA5900/data/'+ str(tics[count]) +'_sim_df.csv', index=False)
#  model.save('/home/zacharyknepp2012/Knepp_OUDSA5900/models/' + str(tics[count]) + 'model')



