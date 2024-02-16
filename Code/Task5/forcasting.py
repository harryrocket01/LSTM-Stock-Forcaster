"""
Forcasting.py
Forcasting and Predicting Stock
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""
# For CPU
import os
import warnings
import logging
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)
tf.get_logger().setLevel("INFO")
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)


# Set Seed
SEED = 84
np.random.seed(SEED)
tf.random.set_seed(SEED)


class Forcasting:
    """
    Class: Forcasting

    This class contains all of the functions needed to suggest
    buy-sell-hold actions for task 6.

    Code is built off Tensor flow and the Keras frame work.
    Inspired by  Jason Brownlee & litritiure and their work
    on time seiresforcasting. Links:

    https://keras.io/api/layers/recurrent_layers/lstm/
    https://bitly.ws/3akBm

    Attributes:
        learningrate (): Learning rate for LSTM
        dropout (): Dropout rate for LSTM
        epochs (): Number of epochs to train lstm
        verbose (): Number of inline prints tensorflow should do
        window (): Size of vector input sliding window
        width (): number of inputed features
        LSTM (): LSTM model

    Methods:
        train_forcaster() : Sets, trains and forcasts stock price
        set_training_set() : sets the training set
        set_hyper_perameters() : sets model hyperperamters
        create_train_window() : converts input training set to sliding window,
        and correct input vector shape
        set_model() : set the model based of provided training set.
        train() : trains model based on hyper perameters
        forcast() : foracsts n number of days
        prediction_analysis() : analysis of forcasting models against ground truth
        foracast_metric(): calculates metrics of forcast

    Args:
        None

    Example:
        forcaster = Forcasting()
    """

    def __init__(self):
        self.learningrate = 0.001
        self.dropout = 0.25
        self.epochs = 10
        self.verbose = 0

        self.window = 15
        self.width = 0
        self.lstm = None

        self.original_dataframe = pd.DataFrame
        self.x_train = []
        self.forcast_start = []
        self.y_train = []

    def train_forcaster(
        self,
        learningrate: float,
        epochs: int,
        dropout: float,
        window: int,
        forcast_depth: int,
        train,
        verbose: int = 0,
    ):
        """
        Function: train_forcaster

        Sets, trains and forcasts forcaster on training set. This
        done using pre built functions below.

        Args:
            learningrate (int): model learning rate
            epochs (int): number of epochs to train
            dr (int): model dropout rate
            window (int): size of shifting window
            forcast_depth (int): how far in the future to predict
            train (): training set to train on
            verbose (int): how mnay inline prints tensorflow should do

        Returns:
            result (array): array of forcast results
        """
        self.set_hyper_perameters(
            learningrate=learningrate,
            epochs=epochs,
            dropout=dropout,
            verbose=verbose,
            window=window,
        )

        self.set_training_set(train)
        self.set_model()
        print(f"Model Set - {self.width} input dimentions")
        self.train()
        print("Model Trained")
        result = self.forcast(forcast_depth)
        print(f"Model Successfuly Forcasted {forcast_depth} days")

        return result

    def set_training_set(self, dataframe):
        """
        Function:set_training_set

        Args:
            dataframe (): Dataframe of training set

        Returns:
            None

        """
        dataframe = dataframe.reset_index(drop=True)
        x_train, y_train = self.create_train_window(dataframe)
        self.width = x_train.shape[2]
        self.x_train = x_train
        self.y_train = y_train

    def set_hyper_perameters(
        self, learningrate=None, epochs=None, dropout=None, verbose=None, window=None
    ):
        """
        Function: set_hyper_perameters

        Sets hyper perameters to what the user want. If nothin
        is set, it does nothing.

        Args:
            learningrate (int): model learning rate
            epochs (int): number of epochs to train
            dropout (int): model dropout rate
            window (int): size of shifting window
        Returns:
            None

        """
        self.epochs = epochs if epochs else self.epochs
        self.dropout = dropout if dropout else self.dropout
        self.learningrate = learningrate if learningrate else self.learningrate
        self.verbose = verbose if verbose else self.verbose
        self.window = window if window else self.window

    def create_train_window(self, dataframe):
        """
        Function: create_train_window

        Creates training set by creating sliding windows of W length.

        E.g if window is 5 days, with 3 features. Each training set point
        will be of size (1, 5, 3).

        Args:
            dataframe (int): dataframe to convert to training set

        Returns:
            x_train (array): x training set
            y_train: y training
        """
        x_train, y_train = [], []
        self.original_dataframe = dataframe
        self.forcast_start = np.array([dataframe.tail(self.window)])

        for i in range(len(dataframe) - self.window - 1):
            current_window = dataframe.iloc[i : i + self.window].to_numpy()
            current_result = dataframe.loc[i + self.window + 1]

            x_train.append(current_window)
            y_train.append(current_result)
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_train = np.reshape(
            x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2])
        )
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

        return x_train, y_train

    def set_model(self):
        """
        Function: set_model

        Sets LSTM model archetecture. This is based off of
        keras suggestions and litiriture. Full description
        is given in the appendix of the report.

        Consists of 3 layers, 2 50 wide LSTM modules, in to
        a dense layer.

        It dynamically scales to the size of provided input
        features.

        Args:
            None
        Returns:
            None
        """
        input_shape = (self.window, self.width)
        model = Sequential()
        model.add(
            LSTM(50, input_shape=input_shape, activation="relu", return_sequences=True)
        )
        model.add(LSTM(50))
        model.add(Dense(self.width))

        model.compile(
            optimizer=Adam(learning_rate=self.learningrate),
            loss="mse",
            metrics=["mean_absolute_error"],
        )

        self.lstm = model

    def train(self):
        """
        Function: train

        Trains LSTM Model on preset training set. This
        is done through pre built TF train function.

        0.1 validation split for each epoch.

        Args:
            None

        Returns:
            None

        """
        history = self.lstm.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=16,
            verbose=self.verbose,
            validation_split=0.1,
        )
        return history

    def forcast(self, forcast_range: int = 20):
        """
        Function: forcast

        Forcasts N number of days in to the future. This
        is done recursively. The model must be trained.

        Args:
            forcast_range (int): number of days to format

        Returns:
            result (array): array of forcasted results

        """
        lstm_input = self.forcast_start
        result = []

        for _ in range(forcast_range):
            prediction = self.lstm.predict(lstm_input, verbose=0)
            result.append(prediction)
            lstm_input = lstm_input[:, 1:, :]
            lstm_input = np.concatenate(
                [lstm_input, prediction.reshape(1, 1, self.width)], axis=1
            )

        result = np.vstack(result)
        result = result[:, 0]

        return result

    def prediction_analysis(self, original_data, real, result_stock, result_aux):
        """
        Function: prediction_analysis

        Creates plots of forcasting, along with generating numeric
        metrics for the forcasting.

        Args:
            original_data (): Dataframe of past stock data
            real (): real data that was predicted
            result_stock (): predictions only using stock data
            result_aux  (): predictions using auxillary data
        Returns:
            None

        """
        dataframe = real.drop(["open", "high", "low", "volume"], axis=1).sort_index(
            ascending=True
        )
        dataframe["result_stock"] = np.array(result_stock)
        dataframe["result_aux"] = np.array(result_aux)

        # Re-scale the forcasted data
        std = original_data["close"].std()
        mean = original_data["close"].mean()
        dataframe["result_stock"] = (dataframe["result_stock"] * std) + mean
        dataframe["result_aux"] = (dataframe["result_aux"] * std) + mean

        last_point = original_data["close"].iloc[-1].copy()
        last_index = original_data["close"].index[-1]

        to_plot = pd.DataFrame(
            {
                "date": last_index,
                "close": [last_point],
                "result_stock": [last_point],
                "result_aux": [last_point],
            }
        ).set_index("date")
        to_plot = pd.concat([to_plot, dataframe.copy()])

        _, axs = plt.subplots(figsize=(5, 3))
        axs.plot(original_data["close"].tail(100))
        axs.plot(mdates.date2num(to_plot.index), to_plot["close"])
        axs.plot(mdates.date2num(to_plot.index), to_plot["result_stock"], "--")
        axs.plot(mdates.date2num(to_plot.index), to_plot["result_aux"], "--")
        axs.legend(
            ["Past Data", "Actual Future", "Model 1 Prediction", "Model 2 Prediction"],
            loc="upper left",
            fontsize="small",
        )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.suptitle("Forcasted Price for AAL stock")
        plt.grid()
        plt.tight_layout()
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.savefig("DAPs_Code/Task5/Graphics/Forcaster Results.pdf")
        plt.close()

        mse_1, mae_1, r_squared_1 = self.foracast_metric(
            dataframe["close"], dataframe["result_stock"]
        )
        print(f"Model 1 Result: MSE: {mse_1} MAE: {mae_1},R2: {r_squared_1}")
        mse_2, mae_2, r_squared_2 = self.foracast_metric(
            dataframe["close"], dataframe["result_aux"]
        )
        print(f"Model 2 Result: MSE: {mse_2} MAE: {mae_2},R2: {r_squared_2}")

        plt.figure(figsize=(3, 3))
        sns.jointplot(
            x=dataframe["close"],
            y=dataframe["result_stock"],
            kind="scatter",
            color="#2ca02c",
        )
        plt.subplots_adjust(top=0.95)
        plt.xlabel("True Values")
        plt.ylabel("Forecasted Values")
        plt.suptitle("Joint Plot of True vs. Model 1 Forecasted Values")
        plt.savefig("DAPs_Code/Task5/Graphics/model1_joint.pdf")
        plt.close()
        plt.figure(figsize=(3, 3))
        sns.jointplot(
            x=dataframe["close"],
            y=dataframe["result_aux"],
            kind="scatter",
            color="#d62728",
        )
        plt.subplots_adjust(top=0.95)
        plt.xlabel("True Values")
        plt.ylabel("Forecasted Values")
        plt.suptitle("Joint Plot of True vs. Model 2 Forecasted Values")
        plt.savefig("DAPs_Code/Task5/Graphics/model2_joint.pdf")
        plt.close()

    def foracast_metric(self, true, predict):
        """
        Function: foracast_metric

        Generates metrics for a given Forcast prediction.

        This include MSE - Mean squared error
        MAE - Mean absolute error
        R squared - coefficient of determination

        Done using Sckit learn

        Args:
            state (int): current state

        Returns:
            mse: Mean square error of prediction
            mae: Mean Absolute error of prediction
            r_squared: r squard value of prediction

        """
        mse = mean_squared_error(true, predict)
        mae = mean_absolute_error(true, predict)
        r_squared = r2_score(true, predict)

        return mse, mae, r_squared
