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
import pandas as pd
pd.set_option('display.max_columns', None)  # Display all columns


#This class takes the previously pulled news data and cleans the text data for analysis
class Text_Cleaner:
    def __init__(self, news_data):

      self.df = news_data
      self.cleaned_text = ''
      self.agg_df = None

      self.bigrams = []
      self.trigrams = []

    # Cleans the text of all unnecessary characters
    def scrub_text(self):

      # Remove rows with NA values
      self.df = self.df.dropna()

      # Define a regex pattern for matching dates in YYYY-MM-DD or MM/DD/YYYY format
      date_pattern = r'\d\d\d\d-\d\d-\d\d'

      dates = self.df['dt']
      new_dates = []
      for i in dates:
        date_found = re.findall(date_pattern, i)
        new_dates.append(str(date_found[0]))

      self.df['clean_dates'] = new_dates

      self.df = self.aggregate_by_date()

      headline_cleaned = []
      headlines = self.df['headline']
      stop_words = set(stopwords.words("english"))
      lemmatizer = WordNetLemmatizer()

      for i in headlines:
        #print(i)

        i.lower()
        tokens = word_tokenize(i)

        # Create bigrams
        bi_grams = list(bigrams(tokens))
        self.bigrams.append(bi_grams)

        # Create trigrams
        tri_grams = list(trigrams(tokens))
        self.trigrams.append(tri_grams)

        # Step 3: Removing HTML tags (if applicable)
        cleaned_tokens = [re.sub(r"<.*?>", "", token) for token in tokens]

        # Step 4: Removing special characters and punctuation
        cleaned_tokens = [re.sub(r"[^a-zA-Z\s]", "", token) for token in cleaned_tokens]

        # Step 5: Removing numbers
        cleaned_tokens = [re.sub(r"\d", "", token) for token in cleaned_tokens]

        # Step 6: Removing stop words
        cleaned_tokens = [token for token in cleaned_tokens if token not in stop_words]

        # Step 7: Lemmatization
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]

        # Step 11: Removing whitespace
        cleaned_tokens = [token.strip() for token in cleaned_tokens]

        # Step 14: Remove empty tokens
        cleaned_tokens = [token for token in cleaned_tokens if token]

        headline_cleaned.append(cleaned_tokens)

        # Step 14: Join cleaned tokens back into a text string
        self.cleaned_text += " ".join(cleaned_tokens)

      self.df['cleaned_text'] = headline_cleaned
      self.calculate_sentiment()
      self.look_for_words()

    def aggregate_by_date(self):
      # Group by 'date_column' and aggregate text using join
      grouped = self.df.groupby('clean_dates').agg({'headline': ' '.join, 'snippet': ' '.join})
      return grouped

    def view_df(self):
      print(self.df)

    def return_df(self):
      return self.df

    def return_cleaned_text(self):
      return self.cleaned_text

    def calculate_sentiment(self):
      head_pos = []
      head_neg = []
      head_neu = []

      cleaned_text_pos = []
      cleaned_text_neg = []
      cleaned_text_neu = []

      for index, row in self.df.iterrows():
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

      self.df['pos'] = head_pos
      self.df['neg'] = head_neg
      self.df['neu'] = head_neu

      self.df['cleaned_pos'] = cleaned_text_pos
      self.df['cleaned_neg'] = cleaned_text_neg
      self.df['cleaned_neu'] = cleaned_text_neu


    def look_for_words(self):
      text = self.df['cleaned_text']
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
            print(war)
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

      self.df['recession'] = recession_list
      self.df['fomc'] = fomc_list
      self.df['inflation'] = inflation_list
      self.df['cpi'] = cpi_list
      self.df['unemployment'] = unemployment_list
      self.df['gdp'] = gdp_list
      self.df['bubble'] = bubble_list
      self.df['bear'] = bear_list
      self.df['bearish'] = bearish_list
      self.df['bull'] = bull_list
      self.df['bullish'] = bullish_list
      self.df['acquires'] = acquires_list
      self.df['acquisition'] = acquisition_list
      self.df['merger'] = merger_list
      self.df['war'] = war_list
      self.df['vix'] = vix_list
      self.df['volatility'] = volatility_list




