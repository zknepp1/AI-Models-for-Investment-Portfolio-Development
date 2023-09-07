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

