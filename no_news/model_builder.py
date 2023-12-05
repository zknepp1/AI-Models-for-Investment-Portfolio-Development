
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, GRU, Dense, Bidirectional
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Bidirectional, Dropout

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Embedding,Dense,LSTM,Dropout,Flatten,BatchNormalization,Input, Attention,Concatenate
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tensorflow.keras.layers import GRU, Dense

# This class will build a neural network for an indivisual stock
class Model_Builder:
    def __init__(self, df):

        self.df = df
        self.y_train = None
        self.y_test = None
        self.x_train = None
        self.x_test = None

        self.y_train_nn = None
        self.y_test_nn = None
        self.x_train_reshaped = None
        self.x_test_reshaped = None

        self.Y = None
        self.X_reshaped = None

        self.X = ['Target','Open', 'High', 'Low','Close','Volume',
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



        self.X_no_target = ['Open', 'High', 'Low','Close','Volume',
                  'five_day_rolling','ten_day_rolling','twenty_day_rolling',
                  'market_open', 'market_high','market_low', 'market_close',
                  'market_volume','market_twenty_roll']



        self.best_model = None
        self.best_mse = None

    # Not usinhg this
    def prep_data(self):
        copy = self.df.copy()

        print(copy.shape)
        copy.dropna(inplace=True)
        print(copy.shape)

        X = copy[self.X_no_target]
        y = copy['Target']

        scaler = StandardScaler()

        # Fit the scaler on the training data
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled)

        self.y_train = y.iloc[:-20]
        self.y_test = y.iloc[-20:]
        self.x_train = X_scaled.iloc[:-20]
        self.x_test = X_scaled.iloc[-20:]



    # Not usinhg this
    def build_rf(self):
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(self.x_train, self.y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print(f"Best Model Parameters: {best_params}")
        print(f"Test Mean Squared Error: {mse}")



    # Separates data frame into train and test 
    # scales data and preps for neural network training
    def train_test_scale(self):
        copy = self.df.copy()
        copy.dropna(inplace=True)

        X = copy[self.X_no_target].values
        Y = copy['Target'].values

        print(X.shape)
        print(Y.shape)

        scaler = StandardScaler()
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        m = 14
        timesteps = 1

        self.X_reshaped = X_scaled.reshape(X_scaled.shape[0], timesteps, m) 
        self.Y = Y

    # Builds the neural network and optimizes best model based on MSE
    def build_and_optimize_models(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        epochs = [100]
        lstm_units1 = [8,64]
        lstm_units2 = [8,64]
        input_shape = (1, 14)


        kfold = KFold(n_splits=5)
        cv_scores = []


        self.best_model = None
        self.best_mse = 1000


        for train_index, test_index in kfold.split(self.X_reshaped):
          x_train_fold, x_val_fold = self.X_reshaped[train_index], self.X_reshaped[test_index]
          y_train_fold, y_val_fold = self.Y[train_index], self.Y[test_index]

          for lstm_unit1 in lstm_units1:
              for lstm_unit2 in lstm_units2:
                  print()
                  print("LSTM unit 1: ", lstm_unit1)
                  print("LSTM unit 2: ", lstm_unit2)
                  #print("LSTM unit 3: ", lstm_unit3)
                  print()

                  # Create a Sequential model
                  model = Sequential()
                  model.add(LSTM(units=lstm_unit1, return_sequences=True, input_shape=input_shape))
                  model.add(Dropout(0.2))
                  model.add(LSTM(units=lstm_unit2, return_sequences=True))
                  model.add(Dropout(0.2))
                  model.add(Dense(units=1, activation='swish'))# the output layer
                  model.compile(optimizer='Adam', loss='mean_squared_error')
                  hist = model.fit(x_train_fold, y_train_fold, epochs=100, validation_data=(x_val_fold, y_val_fold), batch_size=1, verbose=1, callbacks=[early_stopping])
                  predictions = model.predict(x_val_fold)

                  # Assuming predicted_values is a numpy array of shape (num_examples, num_time_steps)
                  average_predictions = np.mean(predictions, axis=1)
                  mse = mean_squared_error(y_val_fold, average_predictions)

                  print("MSE: ", mse)
                  # Determine the number of epochs the model trained for
                  num_epochs_used = len(hist.history['val_loss'])

                  if mse < self.best_mse:
                      self.best_model = model
                      self.best_mse = mse




    def return_best_model(self):
        return self.best_model

    def return_best_mse(self):
        return self.best_mse







