import pandas as pd
import pynytimes
import time
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('vader_lexicon')

pd.set_option('display.max_columns', None)  # Display all columns



#This class is meant to collect news articles from nyt between the years 2000-2023
# The packages used for this class is pynytimes
class News_Collector:
    def __init__(self, start_year, end_year, end_month):
        self.months = list(range(1, end_month))
        self.years = list(range(start_year, end_year))

        self.news_data = self.collect_all_news()
        self.clean_news_data = self.clean_news_data()
        self.sentiment = self.calculate_sentiment()
        self.count_data = self.look_for_words()

    # Collects all news data since the year 2000
    def collect_all_news(self):
        dt = []
        headline = []
        snippet = []
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
                  dt.append(article['pub_date'])
                  headline.append(article['headline']['main'])
                  snippet.append(article['snippet'])

              else:
                print('Error1:', response.status_code)

            except:
              print('Error2:' + str(count))
        df = pd.DataFrame({'dt': dt, 'headline': headline, 'snippet': snippet})
        return df

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
        return self.count_data


    def clean_news_data(self):
        # Remove rows with NA values
        copy = self.news_data.dropna()
        # Define a regex pattern for matching dates in YYYY-MM-DD or MM/DD/YYYY format
        date_pattern = r'\d\d\d\d-\d\d-\d\d'

        dates = copy['dt']
        new_dates = []
        for i in dates:
          date_found = re.findall(date_pattern, i)
          new_dates.append(str(date_found[0]))
        copy['clean_dates'] = new_dates
        grouped = self.aggregate_by_date(copy)
        headline_cleaned = []
        headlines = grouped['headline']
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        for i in headlines:
          i.lower()
          tokens = word_tokenize(i)
          # Create bigrams
          bi_grams = list(bigrams(tokens))
          #self.bigrams.append(bi_grams)
          # Create trigrams
          tri_grams = list(trigrams(tokens))
          #self.trigrams.append(tri_grams)
          # Removing HTML tags (if applicable)
          cleaned_tokens = [re.sub(r"<.*?>", "", token) for token in tokens]
          # Removing special characters and punctuation
          cleaned_tokens = [re.sub(r"[^a-zA-Z\s]", "", token) for token in cleaned_tokens]
          # Removing numbers
          cleaned_tokens = [re.sub(r"\d", "", token) for token in cleaned_tokens]
          # Removing stop words
          cleaned_tokens = [token for token in cleaned_tokens if token not in stop_words]
          # Lemmatization
          cleaned_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]
          # Removing whitespace
          cleaned_tokens = [token.strip() for token in cleaned_tokens]
          # Remove empty tokens
          cleaned_tokens = [token for token in cleaned_tokens if token]
          headline_cleaned.append(cleaned_tokens)

        # Saveing clean tokens to dataframe
        grouped['cleaned_text'] = headline_cleaned
        return grouped


    def aggregate_by_date(self, copy):
        # Group by 'date_column' and aggregate text using join
        grouped = copy.groupby('clean_dates').agg({'headline': ' '.join, 'snippet': ' '.join})
        return grouped

    def calculate_sentiment(self):
        copy = self.clean_news_data
        head_pos = []
        head_neg = []
        head_neu = []

        cleaned_text_pos = []
        cleaned_text_neg = []
        cleaned_text_neu = []

        for index, row in copy.iterrows():
          text1 = row['headline']
          text2 = " ".join(row['cleaned_text'])
          # Create a SentimentIntensityAnalyzer object
          sid = SentimentIntensityAnalyzer()
          # Perform sentiment analysis
          sentiment_scores = sid.polarity_scores(text1)
          head_pos.append(sentiment_scores['pos'])
          head_neg.append(sentiment_scores['neg'])
          head_neu.append(sentiment_scores['neu'])
          sentiment_scores = sid.polarity_scores(text2)
          cleaned_text_pos.append(sentiment_scores['pos'])
          cleaned_text_neg.append(sentiment_scores['neg'])
          cleaned_text_neu.append(sentiment_scores['neu'])

        copy['pos'] = head_pos
        copy['neg'] = head_neg
        copy['neu'] = head_neu
        copy['cleaned_pos'] = cleaned_text_pos
        copy['cleaned_neg'] = cleaned_text_neg
        copy['cleaned_neu'] = cleaned_text_neu
        return copy


    def look_for_words(self):
        text = self.sentiment['cleaned_text']
        copy = self.sentiment

        biden_list = []
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

        for day in text:
          biden = 0
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
            elif i.lower() == 'biden':
              biden += 1
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

          biden_list.append(biden)
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

        copy['biden'] = biden_list
        copy['recession'] = recession_list
        copy['fomc'] = fomc_list
        copy['inflation'] = inflation_list
        copy['cpi'] = cpi_list
        copy['unemployment'] = unemployment_list
        copy['gdp'] = gdp_list
        copy['bubble'] = bubble_list
        copy['bear'] = bear_list
        copy['bearish'] = bearish_list
        copy['bull'] = bull_list
        copy['bullish'] = bullish_list
        copy['acquires'] = acquires_list
        copy['acquisition'] = acquisition_list
        copy['merger'] = merger_list
        copy['war'] = war_list
        copy['vix'] = vix_list
        copy['volatility'] = volatility_list
        copy['rate_cuts'] = rate_cuts_list
        copy['rate_hikes'] = rate_hikes_list
        copy['beat_earnings'] = beat_earnings_list
        copy['beat_eps'] = beat_eps_list
        copy['beat_revenue'] = beat_revenue_list
        copy['missed_earnings'] = missed_earnings_list
        copy['missed_eps'] = missed_eps_list
        copy['missed_revenue'] = missed_revenue_list
        copy['dividend_cut'] = dividend_cut_list
        copy['dividend_raise'] = dividend_raise_list

        return copy





# EXAMPLE CODE
#collector = News_Collector(2020, 2022, 9)
#news_data = collector.return_news_data()
#print(news_data)






