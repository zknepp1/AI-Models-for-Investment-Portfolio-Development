import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from dataframe_collector import DataFrameCollection
from news_collecter import News_Collector
from news_cleanup import Text_Cleaner
from join_data import Join_Data

class Investment_Manager:
    def __init__(self, tickers, start_date):
        self.data = self.collect_recent_financial_data(tickers, start_date)
        self.news_data = self.collect_recent_news()
        self.clean_news_data = self.clean_news_data()
        self.dfs_joined = self.join_dataframes()
        self.AMDmodel = self.retrieve_model()

    def collect_recent_financial_data(self, tickers, start_date):
        collection = DataFrameCollection(tickers, start_date,
                                         datetime.datetime.now().strftime('%Y-%m-%d'))
        collection.retrieve_financial_data()
        df_list = collection.return_dataframes()
        return df_list

    def collect_recent_news(self):
        collector = News_Collector(2022, 2024)
        collector.collect_all_news()
        news_data = collector.return_news_data()
        return news_data

    def clean_news_data(self):
        scrubber = Text_Cleaner(self.news_data)
        scrubber.scrub_text()
        clean_news_data = scrubber.return_df()
        return clean_news_data

    def join_dataframes(self):
        joiner = Join_Data(self.data, self.clean_news_data)
        joiner.combine_dataframes()
        dfs_ready = joiner.loop_time_step_creation()
        for df in dfs_ready:
          df = df.dropna()
        return dfs_ready

    def retrieve_model(self):
        AMD_model = tf.keras.models.load_model('/home/zacharyknepp2012/Knepp_OUDSA5900/models/AMDmodel.h5')
        return AMD_model

    def make_predictions(self):
        copy = self.dfs_joined[0].tail(5)
        print(copy.shape)
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        X = copy.iloc[:, 2:]
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        m = 33
        timesteps = 6
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], timesteps, m)
        preds = self.AMDmodel.predict(X_reshaped)
        print(preds)



tickers = ['AMD', '^GSPC']
l = Investment_Manager(tickers, '2005-1-1')
l.make_predictions()

print('THIS IS THE END...')

