#from keras.optimizers.optimizer import learning_rate_schedule
#from keras.activations import activation_layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, GRU, Dense, Bidirectional
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Embedding,Dense,LSTM,Dropout,Flatten,BatchNormalization,Input, Attention,Concatenate
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tensorflow.keras.layers import GRU, Dense


class Model_Builder:
    def __init__(self, df):

        self.df = df
        self.y_train_nn = None
        self.y_test_nn = None
        self.x_train_reshaped = None
        self.x_test_reshaped = None



        self.X = ['Target','Open', 'High', 'Low','Close','Volume',
                  'five_day_rolling','ten_day_rolling','twenty_day_rolling',
                  'Technology','Blockchain','Economy_Monetary','IPO',
                  'Retail_Wholesale','Financial_Markets','Manufacturing',
                  'Real_Estate','Finance','Life_Sciences','Earnings',
                  'Mergers','Energy','Economy_Fiscal','Economy_Macro',
                  'sentiment_labels_Bullish','sentiment_labels_Bearish',
                  'sentiment_labels_Neutral','sentiment_labels_Somewhat-Bullish',
                  'sentiment_labels_Somewhat-Bearish',
                  'market_open', 'market_high','market_low', 'market_close',
                  'market_volume','market_twenty_roll']



        self.best_model = None
        self.best_mse = None


    def train_test_scale(self):
        copy = self.df.copy()

        print(copy.shape)
        copy.dropna(inplace=True)
        print(copy.shape)

        train = copy.iloc[:-20]
        test = copy.iloc[-20:]

        print('Train shape: ', train.shape)
        print('Test shape: ', test.shape)

        X_train = train[self.X].values
        Y_train = train['Target'].values

        X_test = test[self.X].values
        Y_test = test['Target'].values

        # Convert to numpy arrays
        x_train_nn = np.array(X_train)
        y_train_nn = np.array(Y_train)
        x_test_nn = np.array(X_test)
        y_test_nn = np.array(Y_test)
        self.y_train_nn = y_train_nn
        self.y_test_nn = y_test_nn

        # Assuming you have split your data into X_train and X_test
        scaler = StandardScaler()

        # Fit the scaler on the training data
        scaler.fit(X_train)

        # Transform both training and test data using the same scaler
        X_train_scaled = scaler.transform(x_train_nn)
        X_test_scaled = scaler.transform(x_test_nn)

        # Example data: n samples, each with m features
        m = 35
        timesteps = 1

        # Reshape the data to match neural network input shape
        self.x_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], timesteps, m)  # Adding an extra dimension for single feature
        self.x_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], timesteps, m)  # Adding an extra dimension for single feature


    def build_and_optimize_models(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        epochs = [100]
        lstm_units1 = [2]
        lstm_units2 = [2]
        lstm_units3 = [128]
        input_shape = (1, 35)

        model = Sequential()
        model.add(LSTM(units=10, return_sequences=False, input_shape=input_shape))
        model.add(Dense(units=10, activation='swish'))
        model.add(Dense(units=1, activation='swish'))
        model.compile(optimizer='Adam', loss='mean_squared_error')
        hist = model.fit(self.x_train_reshaped, self.y_train_nn, epochs=10, batch_size=2, verbose=1)
        predictions = model.predict(self.x_test_reshaped)
        mse = mean_squared_error(self.y_test_nn, predictions)

        self.best_model = model
        self.best_mse = mse

        for lstm_unit1 in lstm_units1:
            for lstm_unit2 in lstm_units2:
                for lstm_unit3 in lstm_units3:
                    print()
                    print("LSTM unit 1: ", lstm_unit1)
                    print("LSTM unit 2: ", lstm_unit2)
                    print("LSTM unit 3: ", lstm_unit3)
                    print()

                    # Create a Sequential model
                    model = Sequential()
                    model.add(LSTM(units=lstm_unit1, return_sequences=True, input_shape=input_shape))
                    model.add(Dropout(0.2))
                    model.add(LSTM(units=lstm_unit2, return_sequences=True))
                    model.add(Dropout(0.2))
                    model.add(LSTM(units=lstm_unit3, return_sequences=True))
                    model.add(Dropout(0.2))
                    model.add(Dense(units=100, activation='swish'))
                    model.add(Dense(units=1, activation='swish'))# the output layer
                    model.compile(optimizer='Adam', loss='mean_squared_error')
                    hist = model.fit(self.x_train_reshaped, self.y_train_nn, epochs=100, validation_data=(self.x_test_reshaped, self.y_test_nn), batch_size=1, verbose=1, callbacks=[early_stopping])
                    predictions = model.predict(self.x_test_reshaped)

                    # Assuming predicted_values is a numpy array of shape (num_examples, num_time_steps)
                    average_predictions = np.mean(predictions, axis=1)
                    mse = mean_squared_error(self.y_test_nn, average_predictions)

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
