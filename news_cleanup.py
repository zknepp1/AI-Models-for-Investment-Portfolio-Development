#This class takes the previously pulled news data and cleans the text data for analysis
class Text_Cleaner:
    def __init__(self):

      self.df = pd.read_csv('/content/drive/My Drive/news/newsdata.csv')
      self.cleaned_text = ''


    # Cleans the text of all unnecessary characters
    def scrub_text(self):

      # Remove rows with NA values
      self.df = self.df.dropna()

      headline_cleaned = []
      headlines = self.df['headline']
      stop_words = set(stopwords.words("english"))
      lemmatizer = WordNetLemmatizer()

      count = 0

      for i in headlines:
        #print(i)

        i.lower()
        tokens = word_tokenize(i)

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

        headline_cleaned.append(cleaned_tokens)

        # Step 14: Join cleaned tokens back into a text string
        self.cleaned_text += " ".join(cleaned_tokens)

        count += 1
        print(count)

      self.df['cleaned_text'] = headline_cleaned



    def view_df(self):
