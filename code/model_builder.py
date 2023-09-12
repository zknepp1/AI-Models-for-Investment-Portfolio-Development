from keras.optimizers.optimizer import learning_rate_schedule
from keras.activations import activation_layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, GRU, Dense, Bidirectional


class Model_Builder:
    def __init__(self, df):
        self.df = df
        self.y_train_nn = None
        self.y_test_nn = None
        self.x_train_reshaped = None
        self.x_test_reshaped = None

        self.best_model = None
        self.best_mse = None


    def train_test_scale(self):
        copy = dfs_ready[0].copy()
        copy.dropna(inplace=True)

        train = copy.iloc[:-90]
        test = copy.iloc[-90:]

        X_train = train.iloc[:, 1:].values
        Y_train = train['Target'].values
        X_test = test.iloc[:, 1:].values
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
        m = 18
        timesteps = 6

        # Reshape the data to match neural network input shape
        self.x_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], timesteps, m)  # Adding an extra dimension for single feature
        self.x_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], timesteps, m)  # Adding an extra dimension for single feature


    def build_and_optimize_models(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        epochs = [100]
        lstm_units = [5, 50]
        #optimizers = ['Adam','RMSprop']
        optimizers = ['Adam']

        input_shape = (6, 18)

        model = Sequential()
        model.add(LSTM(units=100, return_sequences=False, input_shape=input_shape))
        dense_units = 1  # Number of units in the output layer
        model.add(Dense(units=500, activation='swish'))
        model.add(Dense(units=dense_units, activation='swish'))
        model.compile(optimizer='Adam', loss='mean_squared_error')
        hist = model.fit(self.x_train_reshaped, self.y_train_nn, epochs=5, batch_size=32, verbose=1)
        predictions = model.predict(self.x_test_reshaped)
        mse = mean_squared_error(self.y_test_nn, predictions)

        print("MSE: ", mse)

        self.best_model = model
        self.best_mse = mse

        for epoch in epochs:
            for opt in optimizers:
                for lstm_unit in lstm_units:
                    print()
                    print("Epochs: ", epoch)
                    print('Optimizer: ', opt)
                    print("Batch size: ", 1)
                    print("LSTM units: ", lstm_unit)
                    print()

                    # Create a Sequential model
                    model = Sequential()
                    #model.add(LSTM(units=lstm_unit, return_sequences=False, input_shape=input_shape))
                    model.add(Bidirectional(LSTM(units=lstm_unit, return_sequences=False), input_shape=input_shape))
                    #model.add(Bidirectional(LSTM(units=lstm_unit, return_sequences=False)))
                    model.add(Dense(units=1, activation='swish'))# the output layer
                    model.compile(optimizer=opt, loss='mean_squared_error')
                    hist = model.fit(self.x_train_reshaped, self.y_train_nn, epochs=epoch, validation_data=(self.x_test_reshaped, self.y_test_nn), batch_size=1, verbose=1, callbacks=[early_stopping])
                    predictions = model.predict(self.x_test_reshaped)
                    mse = mean_squared_error(self.y_test_nn, predictions)

                    print("MSE: ", mse)
                    # Determine the number of epochs the model trained for
                    num_epochs_used = len(hist.history['val_loss'])

                    # Apply dropout after determining the optimal number of epochs
                    model.add(Dropout(0.5))  # Example dropout rat

                    self.plot_predictions(predictions)



                    if mse < self.best_mse:
                        self.best_model = model
                        self.best_mse = mse


    def plot_predictions(self, preds):
        plt.plot(preds, label='Predictions')
        plt.plot(self.y_test_nn, label='Actual')
        plt.legend()
        plt.show()




    def plot_best_model_results(self):
        preds = self.best_model.predict(self.x_test_reshaped)
        plt.plot(preds, label='Predictions')
        plt.plot(self.y_test_nn, label='Actual')
        plt.legend()
        plt.show()



    def return_best_model(self):
        return self.best_model

    def return_best_mse(self):
        return self.best_mse
