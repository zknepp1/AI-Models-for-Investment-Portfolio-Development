import datetime


from dataframe_collector import DataFrameCollection
from news_collecter import News_Collector
from news_cleanup import Text_Cleaner
from join_data import Join_Data




class Investment_Manager:
    def __init__(self, tickers, start_date):
        self.data = self.collect_recent_financial_data(tickers, start_date)
        self.news_data = self.collect_recent_news()
        self.clean_news_data = self.clean_news_data()
        #self.dfs_joined = self.join_dataframes()


    def collect_recent_financial_data(self, tickers, start_date):
        collection = DataFrameCollection(tickers, start_date,
                                         datetime.datetime.now().strftime('%Y-%m-%d'))
        collection.retrieve_financial_data()
        collection.save_data()
        df_list = collection.return_dataframes()
        return df_list

    def collect_recent_news(self):
        collector = News_Collector(2022, 2023)
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
        joiner.pop_market_df()
        joiner.combine_dataframes()
        dfs_ready = joiner.loop_time_step_creation()
        return dfs_ready


tickers = ['AMD', '^GSPC']
l = Investment_Manager('AMD', '2005-1-1')
print()
print()
print()
#print(l.dfs_joined[0])
print()
print()


print('THIS IS THE END...')

