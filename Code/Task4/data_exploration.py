"""
DataExploration.py
Forcasting and Predicting Stock
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns
from scipy.stats import ranksums


class DataExploration:
    """
    Class: DataExploration

    Class containing all the exploration for Task 4. It contains untilities
    for plotting and analysis of the data.

    Attributes:
        dataframe (pd.Dataframe): copy of dataframe to process
        root (string): root where to save graphics

    Methods:
        derrive_insights(): runs array of functions within sequential order
        ohcl(): Creates OHCL plot for monthly data
        seasonality() Creates plots exploring seasonality
        distribution() Creates a plot looking at data distribution
        correlation(): Creates corrilation matrix
        hypothesis(): Performs hypothesis testing

    Args:
        data (pd.DataFrame): Input dataframe containing all the data to process

    Example:
        explore = DataExploration(dataframe)
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe.copy()
        self.root = "DAPs_Code/Task4/Graphics/"

    def derrive_insights(self):
        """
        Function: derrive_insights

        Derrives insites of the data using the following functions, creating
        plots of them all. Done by running each function in the desired
        order.

        Includes seasonality, correlation, distribution and hypothesis
        testing.

        Args:
            None
        Returns:
            None
        Example:
            explore = DataExploration(dataframe).derrive_insights()
        """
        self.ohcl()
        print("Created OHCL Plot")
        self.seasonality()
        # self.polar_plot("holiday", "Yearly polar plot for Holiday trend Data")
        self.polar_plot(
            "temperature_2m_mean", "Yearly polar plot for 2m Mean Temperature"
        )
        self.polar_plot("close", "Yearly polar plot for Closing Price of AAL")
        self.polar_plot("volume", "Yearly polar plot for Volume of AAL")
        self.polar_plot("value_CPI", "Yearly polar plot for CPI")

        print("Performed Seasonal Decomposition")
        self.correlation()
        print("Calculated and plotted Correlation Matrix")
        self.distribution()
        print("Produced Distribution Histogram Plots")

        self.granger_test("close", "value_WTI")
        self.granger_test("close", "volume")
        self.granger_test("close", "value_CPI")
        self.granger_test("close", "temperature_2m_mean")
        self.granger_test("close", "AAL")

    def ohcl(self):
        """
        Function: ohcl

            Creates a OHCL plot of stock data, on a monthly scale. Daily and weekly is too messy.
            Shows high, low, open, close, and volume over two plots. Also shows overall trend and
            change within the month. Plot is saved.

        Args:
            None

        Returns:
            None

        Example:
            explore = DataExploration(dataframe).ohcl()
        """
        _, axs = plt.subplots(2, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        grouped_data = self.dataframe.groupby(pd.Grouper(freq="M"))

        monthly_data = {
            "month_data": [],
            "high": [],
            "low": [],
            "open": [],
            "close": [],
            "volume": [],
        }

        for month, month_data in grouped_data:
            monthly_data["month_data"].append(month)
            monthly_data["high"].append(month_data["high"].max())
            monthly_data["low"].append(month_data["low"].min())
            monthly_data["open"].append(month_data["open"].iloc[0])
            monthly_data["close"].append(month_data["close"].iloc[-1])
            monthly_data["volume"].append(month_data["volume"].sum())

        monthly_data = pd.DataFrame(monthly_data)
        monthly_data["pd_1"] = monthly_data["close"] - monthly_data["open"]
        monthly_data["pd_2"] = monthly_data["high"] - monthly_data["close"]
        monthly_data["pd_3"] = monthly_data["low"] - monthly_data["open"]

        axs[0].bar(
            monthly_data["month_data"],
            monthly_data["pd_1"],
            width=20,
            bottom=monthly_data["open"],
            color=["#2ca02c" if x >= 0 else "#d62728" for x in monthly_data["pd_1"]],
            zorder=2,
        )
        axs[0].bar(
            monthly_data["month_data"],
            monthly_data["pd_2"],
            width=5,
            bottom=monthly_data["close"],
            color="black",
            zorder=1,
        )
        axs[0].bar(
            monthly_data["month_data"],
            monthly_data["pd_3"],
            width=5,
            bottom=monthly_data["open"],
            color="black",
            zorder=1,
        )

        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        axs[0].set_ylabel("Price")
        axs[0].set_title("OHLC Chart - AAL")

        axs[1].bar(monthly_data["month_data"], monthly_data["volume"], width=20)
        axs[1].set_ylabel("Volume")
        axs[1].set_xlabel("Date")

        plt.savefig("DAPs_Code/Task4/Graphics/OHLC.pdf")
        plt.close()

    def seasonality(self):
        """
        Function: seasonality

        Performs seasonal decomposition on each column of the dataframe.
        This is done through the existing seasonal_decompose function
        within python. Produces a plot and saves all the plots locally.

        Args:
            None
        Returns:
            None
        Example:
            explore = DataExploration(dataframe).seasonality()
        """
        column_names = self.dataframe.columns

        for current in column_names:
            result = seasonal_decompose(
                self.dataframe[current], model="additive", period=5
            )

            fig, axs = plt.subplots(4, sharex=True)

            fig.suptitle("Trend and Seasonality of " + current)
            axs[0].plot(self.dataframe[current], label="Original Time Series")
            axs[0].legend()
            axs[1].plot(result.trend, label="Trend")
            axs[1].legend()
            axs[2].plot(result.seasonal, label="Seasonal")
            axs[2].legend()
            axs[3].plot(result.resid, label="Residual")
            axs[3].legend()

            plt.tight_layout()
            plt.savefig(
                f"DAPs_Code/Task4/Graphics/Trend + Seasonality/seasonality_{current}.pdf"
            )
            plt.close()

    def polar_plot(self, column: str, title: str):
        """
        Function: polar_plot

        Plots data on Polar Plot to show seasonality.

        Args:
            column (str): name of selected column
            title (str): title for plot
        Returns:
            None
        Example:
            explore = DataExploration(dataframe).polar_plot()
        """
        _, axs = plt.subplots(subplot_kw={"projection": "polar"})

        theta = self.dataframe.index.dayofyear * (2 * np.pi) / 365.25

        cmap = plt.get_cmap("viridis")

        for i in range(len(theta) - 1):
            color = cmap(i / len(theta))
            axs.plot(
                [theta[i], theta[i + 1]],
                [self.dataframe[column].iloc[i], self.dataframe[column].iloc[i + 1]],
                color=color,
            )
        axs.set_xticks([])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"DAPs_Code/Task4/Graphics/polar_{column}.pdf")
        plt.close()

    def distribution(self):
        """
        Function: distribution

        Creates histograms to show the distribution of the data. Plots
        are produced and saved.

        Args:
            None
        Returns:
            None
        Example:
            explore = DataExploration(dataframe).distribution()
        """
        column_names = self.dataframe.columns

        for current in column_names:
            plt.figure(figsize=(7, 3))
            plt.subplot(1, 2, 1)
            sns.histplot(
                self.dataframe[current],
                kde=True,
                label=current,
                element="step",
                stat="density",
            )
            plt.xlabel("Value")
            plt.ylabel("Frequency (%)")
            plt.title("Distribution of " + current)

            differences = np.diff(self.dataframe[current])

            plt.subplot(1, 2, 2)
            sns.histplot(
                differences,
                kde=True,
                element="step",
                stat="density",
            )

            plt.xlabel("Value")
            plt.ylabel("Frequency (%)")
            plt.title("Distribution of 1st Derrivative")
            plt.tight_layout()
            plt.savefig(
                f"DAPs_Code/Task4/Graphics/Distribution/distribution_{current}.pdf"
            )
            plt.close()

    def correlation(self):
        """
        Function: correlation

        Creates a corrilation matrix of all of the columns to explore the
        data correlation between each other. A plot is produced and
        locally saved.

        Args:
            None
        Returns:
            None
        Example:
            explore = DataExploration(dataframe).seasonality()
        """
        correlation_matrix = self.dataframe.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig("DAPs_Code/Task4/Graphics/correlation.pdf")  #
        plt.close()

    def mwu_test(self, column1, column2, alpha=0.05):
        """
        Function: mwu_test

        Performs hypothesis testing using the Mann Whity U Test.

        Args:
            column1 (str): name of column 1 to test
            column2 (str): name of column2 to test
            lag (int): lag to computer test over
        Returns:
            None
        Example:
            explore = DataExploration(dataframe).hypothesis()
        """
        print(f"\nMann Whity U Test - {column1} and {column2}")
        group1 = self.dataframe[column1]
        group2 = self.dataframe[column2]

        u_stat, p_value = ranksums(group1, group2)

        reject_null = p_value < alpha

        if reject_null:
            print(f"Reject Null Hypothesis - {p_value}")
        else:
            print("Do not Reject Null")

        return u_stat, p_value, reject_null

    def granger_test(
        self, column1: str, column2: str, lag: int = 15, threshold: float = 0.3
    ):
        """
        Function:

        Performs granger causality test on two columns. This test
        is to see if there is any caustaility between two variables,
        with a lag of n days.

        Args:
            column1 (str): name of column 1 to test
            column2 (str): name of column2 to test
            lag (int): lag to computer test over
            threshold(float): threshold to reject null hypotheis
        Returns:
            None
        Example:
            explore = DataExploration(dataframe).granger_test()
        """
        print(f"\nGranger Causality Test - {column1} and {column2}")
        to_test = self.dataframe[[column1, column2]].copy()
        result = grangercausalitytests(to_test, lag, verbose=False)

        smallest_p = 1
        lag_val = 0
        for i in range(1, lag + 1):
            p_value = result[i][0]["ssr_chi2test"][1]
            lag = result[i][0]["ssr_chi2test"][2]
            if smallest_p > p_value:
                smallest_p = p_value
                lag_val = lag

        if smallest_p < threshold:
            print(f"Reject Null Hypothesis - {smallest_p}  {lag_val}")
            return True
        print(f"Do not reject Null Hypothesis - {smallest_p}  {lag_val}")
        return False


class FinIndicators:
    """
    Class: FinIndicators

    A class that creates finantial indicators based off the finantial data.
    This class can calculate ADX,MACD and RSI

    Attributes:
        dataframe (pd.Dataframe): copy of dataframe to process
        financial_indicators (pd.DataFrame): dataframe to contain data
        of the finantial indicators.

    Methods:
        RSI() : Calcualtes RSI
        MACD() : Calculates MACD and MACD singal
        ADX() : Caclulates ADX
        calculate_indicators() : Calculates all three, and plots them.

    Args:
        dataframe (pd.DataFrame): Input dataframe containing all the data to process

    Example:
        explore = FinIndicators(dataframe)
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.financial_indicators = pd.DataFrame(
            {
                "rsi": [None] * len(dataframe),
                "macd": [None] * len(dataframe),
                "macd_signal": [None] * len(dataframe),
                "adx": [None] * len(dataframe),
            },
            index=self.dataframe.index,
        )

    def rsi(self, window_size: int = 10):
        """
        Function: RSI

        Calculates Relative Strength Index (RSI) based off of the
        closing prices of the stock on a daily basis.

        Implemntation based off of:
        https://www.investopedia.com/terms/r/rsi.asp

        Args:
            window_size (int): window size to smooth RSI over
        Returns:
            None
        Example:
                explore = FinIndicators(dataframe).RSI()
        """
        difference = np.diff(self.dataframe["close"])
        gains, losses = [], []
        for diff in difference:
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))

        rsi_list = []
        ag_list = []
        al_list = []

        for day in range(len(difference) + 1):
            start = max(day - window_size, 0)
            average_gain = sum(gains[start:day]) / window_size
            average_loss = sum(losses[start:day]) / window_size

            ag_list.append(average_gain)
            al_list.append(average_loss)
            if day < window_size:
                # step 1
                rsi_1 = 100 - (100 / (1 + (average_gain / (average_loss + 0.000001))))
                rsi_list.append(rsi_1)
            else:
                # step 2
                rsi_2 = 100 - (
                    100
                    / (
                        1
                        + (
                            ((ag_list[-1] * (window_size - 1)) + average_gain)
                            / ((al_list[-1] * (window_size - 1)) + average_loss)
                        )
                    )
                )
                rsi_list.append(rsi_2)

        self.financial_indicators["rsi"] = rsi_list

    def macd(self, fast_period: int = 12, slow_period: int = 26, span: int = 9):
        """
        Function: MACD

        Calculates Moving Average Convergence/Divergence (MACD) and
        MACD Signal based of the closing price on a daily basis.

        Implemntation based off of:
        https://www.investopedia.com/terms/m/macd.asp

        Args:
            fast_period (int): number of days to calculated the fast period over
            slow_period (int): number of days to calculated the slow period over
            span (int): days to smooth over
        Returns:
            None
        Example:
            explore = FinIndicators(dataframe).MACD()
        """
        fast = self.dataframe["close"].ewm(span=fast_period, adjust=False).mean()
        slow = self.dataframe["close"].ewm(span=slow_period, adjust=False).mean()
        macd = fast - slow
        # use setting of 26 12 and 9 days
        self.financial_indicators["macd"] = macd
        self.financial_indicators["macd_signal"] = macd.ewm(
            span=span, adjust=False
        ).mean()

    def adx(self, window: int = 14):
        """
        Function: ADX

        Calculates average directional index (ADX) based of the closing, high
        and low price on a daily basis. This function takes a few days before
        correctly converging.

        Implemntation based off of:
        https://www.investopedia.com/articles/trading/07/adx-trend-indicator.asp

        Args:
            window (int): window to smooth ADX over.
        Returns:
            None
        Example:
            explore = FinIndicators(dataframe).ADX()
        """
        adx_copy = self.dataframe.copy()
        adx_copy.reset_index(inplace=True)

        adx_copy["um"] = adx_copy["high"] - adx_copy["high"].shift(1)
        adx_copy["dm"] = adx_copy["low"].shift(1) - adx_copy["low"]

        adx_copy["p_dm"] = np.where(
            (adx_copy["um"] > adx_copy["dm"]) & (adx_copy["um"] > 0), adx_copy["um"], 0
        )
        adx_copy["n_dm"] = np.where(
            (adx_copy["dm"] > adx_copy["um"]) & (adx_copy["dm"] > 0), adx_copy["dm"], 0
        )

        adx_copy["hl"] = adx_copy["high"] - adx_copy["low"]
        adx_copy["hc"] = adx_copy["high"] - adx_copy["close"].shift(1)
        adx_copy["lc"] = adx_copy["low"] - adx_copy["close"].shift(1)

        adx_copy["hl"] = np.abs(adx_copy["hl"])
        adx_copy["hc"] = np.abs(adx_copy["hc"])
        adx_copy["lc"] = np.abs(adx_copy["lc"])

        adx_copy["tr"] = adx_copy[["hl", "lc", "hc"]].max(axis=1)

        adx_copy["p_dm_smooth"] = adx_copy["p_dm"].rolling(window=window).mean()
        adx_copy["n_dm_smooth"] = adx_copy["n_dm"].rolling(window=window).mean()
        adx_copy["tr_smooth"] = adx_copy["tr"].rolling(window=window).mean()

        adx_copy["p_di"] = (100 * adx_copy["p_dm_smooth"]) / adx_copy["tr_smooth"]
        adx_copy["n_di"] = (100 * adx_copy["n_dm_smooth"]) / adx_copy["tr_smooth"]

        adx_copy["dx"] = (100 * (adx_copy["p_di"] - adx_copy["n_di"]).abs()) / (
            adx_copy["p_di"] + adx_copy["n_di"]
        ).abs()

        adx_copy["adx"] = adx_copy["dx"].rolling(window=window).mean()

        for i in range(window * 2 - 1, len(adx_copy)):
            adx_copy.loc[i, "adx"] = (
                (adx_copy.loc[i - 1, "adx"] * (window - 1)) + adx_copy.loc[i, "dx"]
            ) / window

        adx_copy.set_index("date", inplace=True)
        self.financial_indicators["adx"] = adx_copy["adx"]

    def calculate_indicators(self):
        """
        Function: calculate_indicators

        Calculates RSI MACD and ADX, creates a plot of them, and
        returns the calulated values within a indapendant dataframe.

        Args:
            None
        Returns:
            self.financial_indicators (pd.DataFrame): Dataframe of calculated indicators
        Example:
            explore = FinIndicators(dataframe).ADX()
        """
        self.rsi()
        self.macd()
        self.adx()
        print("Calculated RSI MACD and ADX")
        fig, axs = plt.subplots(3, sharex=True)
        x_axis = self.dataframe.index.tolist()
        axs[0].plot(x_axis, self.financial_indicators["rsi"])
        fig.suptitle("Derrived Finantial Indicators of AAL")
        axs[0].set_ylabel("RSI")
        axs[1].plot(x_axis, self.financial_indicators["macd"])
        axs[1].plot(x_axis, self.financial_indicators["macd_signal"])
        axs[1].set_ylabel("MACD")
        axs[1].legend(["MACD", "Signal"])
        axs[2].plot(x_axis, self.financial_indicators["adx"])
        axs[2].set_xlabel("Date")
        axs[2].set_ylabel("ADX")
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        fig.set_size_inches(4, 5)
        plt.tight_layout()
        plt.savefig("DAPs_Code/Task4/Graphics/financial_idicators.pdf")
        plt.close()

        return self.financial_indicators
