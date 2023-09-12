import pandas as pd
import pynytimes
import time
import requests

#This class is meant to collect news articles from nyt between the years 2000-2023
# The packages used for this class is pynytimes
class News_Collector:
    def __init__(self, start_year, end_year):
        self.months = list(range(1, 13))
        self.years = list(range(start_year, end_year))

        self.dt = []
        self.headline = []
        self.snippet = []


    # Collects all news data since the year 2000
    def collect_all_news(self):
        count = 0
        for i in range(len(self.years)):
          for j in range(len(self.months)):
            try:
              time.sleep(15)
              base_url = 'https://api.nytimes.com/svc/archive/v1/' + str(self.years[i]) + '/' + str(self.months[j]) + '.json?api-key=9WZV42GGGa7VnNznPal0BZD427T2KJQC'
              # Make the API request
              response = requests.get(base_url)
              # Check if the request was successful
              if response.status_code == 200:
                data = response.json()
                # Extract and print article headlines and snippets
                for article in data['response']['docs']:
                  self.dt.append(article['pub_date'])
                  self.headline.append(article['headline']['main'])
                  self.snippet.append(article['snippet'])

              else:
                print('Error1:', response.status_code)

            except:
              print('Error2:' + str(count))



    # A function where you can query for a specific topic in nyt database
    def query_news_data(self, topic):
        apikey = "qMNbG2buoVVqqAu7ygcTkE4dBtx9x1l8"
        nytapi = pynytimes.NYTAPI(apikey, parse_dates=True)

        # searching for specific articles
        articles = nytapi.article_search(query = topic, results = 5000, options = {
          "sort": "relevance"}, dates = {"begin": datetime.datetime(2000, 1, 1),"end": datetime.datetime(2023, 12, 31)})


    # Saves the dataframe locally
    def save_all_news_data(self):
        df = pd.DataFrame({'dt': self.dt, 'headline': self.headline, 'snippet': self.snippet})
        df.to_csv('/content/drive/My Drive/news/newsdata.csv')
        print('Saved data to newsdata.csv')


    def return_news_data(self):
        df = pd.DataFrame({'dt': self.dt, 'headline': self.headline, 'snippet': self.snippet})
        return df



