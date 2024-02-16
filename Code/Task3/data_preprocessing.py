"""
DataPreProcessing.py
Forcasting and Predicting Stock
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import zscore
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DataPreProcessing:
    """
    Class: DataPreProcessing

    DataPreProcessing handels task 3 of the project. This task involves handling NaN datapoints,
    detecting and handling outliers, processing these outliers. It addtionally produces
    an array of graphics found within the Graphics files within this folder. Finally
    it transforms the data using normalsiation and PCA dimensionality reduction.

    Attributes:
        dataframe (pd.DataFrame): Dataframe containing all the desired columns to process
        column_name (ArrayLike): Array of column names

    Methods:
        pre_process() : Runs array of preprocessing functions sequentially
        weather_processing() : Processes weather columns
        interpolate() : detected NaN and interpolates values
        outlier_processing(): Detectes outliers and interpolates them
        normalise() : normalises data for desired technique
        pca_test() : performs pca testing and dimentionality reduction
        data_transformation() : transforms data using normalsiation and Dimention reduction

    Args:
        data (pd.DataFrame): Input dataframe containing all the data to process

    Example:
        instance = DataPreProcessing(dataframe)
    """

    def __init__(self, data: pd.DataFrame):
        self.dataframe = data
        self.dataframe["date"] = pd.to_datetime(self.dataframe["date"])
        self.dataframe = self.dataframe.set_index("date")
        self.column_name = self.dataframe.columns

    def pre_process(self):
        """
        Function: pre_process

        this function groups all the preprocessing steps together, to execute them
        sequentually.

        Args:
            None

        Returns:
            self.dataframe (pd.DataFrame): Updated and processed dataframe of the data

        Example:
            processed_data = DataPreProcessing(dataframe).pre_process()
        """
        self.plot_variable()
        self.weather_processing()
        print("Processed Weather")
        self.interpolate()
        print("Interpolated Missing Points")
        self.outlier_processing()
        print("Clamped Outliers")
        self.plot_variable("Processed")

        self.dataframe = self.dataframe.sort_index(ascending=True)
        return self.dataframe.copy()

    def plot_variable(self, route: str = "Raw"):
        """
        Function: plot_raw

        Plots the time sereis data on a plot.

        Args:
            route (string): where to save

        Returns:
            None

        Example:
            processed_data = DataPreProcessing(dataframe).plot_raw()
        """
        for column in self.column_name:
            _, axs = plt.subplots(figsize=(5, 3))
            self.dataframe[column].plot()
            plt.title("Time seires plot of column " + str(column))
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.grid()
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.tight_layout()
            plt.savefig(f"DAPs_Code/Task3/Graphics/{route}/{column}.pdf")
            plt.close()

    def weather_processing(self):
        """
        Function: weather_processing

        processes the weather data. This takes all of the 10 airport data, and combines
        them in to a single column. Updates dataframe stored within class.

        Args:
            None

        Returns:
            None

        Example:
            processed_data = DataPreProcessing(dataframe).weather_processing()
        """
        weather_columns = [
            "weather_code",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
        ]

        for data_type in weather_columns:
            temporary_df = pd.DataFrame(index=self.dataframe.index)
            columns = [col for col in self.dataframe.columns if data_type in col]
            _, axs = plt.subplots(figsize=(5, 3))
            for item in columns:
                axs.plot(
                    self.dataframe.index.to_list(), self.dataframe[item], alpha=0.4
                )

            for i, _ in self.dataframe.iterrows():
                values = []
                for selected_column in columns:
                    values.append(self.dataframe.at[i, selected_column])

                if data_type == "weather_code":
                    temporary_df.at[i, data_type] = max(set(values), key=values.count)
                else:
                    temporary_df.at[i, data_type] = sum(values) / len(values)

            axs.plot(temporary_df[data_type], color="black")
            axs.set_title("Processed data for " + str(data_type))
            axs.set_xlabel("Date")
            axs.set_ylabel("Value")

            axs.legend(
                [
                    "CLT",
                    "ORD",
                    "DFW",
                    "LAX",
                    "MIA",
                    "JFK",
                    "LGA",
                    "PHL",
                    "PHX",
                    "DCA",
                    "Aggragate",
                ],
                fontsize="small",
            )
            plt.grid()
            plt.tight_layout()
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.savefig(f"DAPs_Code/Task3/Graphics/Weather/{data_type}.pdf")
            plt.close()

            self.dataframe = self.dataframe.drop(columns, axis=1).join(temporary_df)
        self.column_name = self.dataframe.columns

    def interpolate(self, interpol_dict: dict = {}):
        """
        Function: interpolate

        Detectes and interpolates missing data points. For data points,
        that can be interpolated with the deisred value they are linerarly extended.
        It produces plots and saves them in the Graphics.Interpolate file.

        Updates dataframe stored within class.

        Args:
            interpol_dict (dict): dictionary containing desired interpolation technique

        Returns:
            None

        Example:
            processed_data = DataPreProcessing(dataframe).interpolate()
        """
        for column in self.column_name:
            _, axs = plt.subplots(figsize=(5, 3))
            axs.scatter(
                self.dataframe.index,
                self.dataframe[column],
                s=6,
                color="#d62728",
                zorder=2,
            )

            if self.dataframe[column].isnull().any().any():
                self.dataframe = self.dataframe.reset_index()
                try:
                    tecnique = interpol_dict[column]
                except Exception:
                    tecnique = "cubicspline"

                match tecnique:
                    case "linear":
                        self.dataframe[column] = self.dataframe[column].interpolate(
                            method="linear"
                        )
                    case "poly":
                        self.dataframe[column] = self.dataframe[column].interpolate(
                            method="polynomial", order=5
                        )
                    case "cubicspline":
                        self.dataframe[column] = self.dataframe[column].interpolate(
                            method="cubicspline"
                        )
                    case "krogh":
                        self.dataframe[column] = self.dataframe[column].interpolate(
                            method="krogh"
                        )
                    case _:
                        pass
                self.dataframe = self.dataframe.set_index("date")

            axs.plot(self.dataframe[column], color="#1f77b4", zorder=1)
            axs.set_title("Interpolated data for " + str(column))
            axs.set_xlabel("Date")
            axs.set_ylabel("Value")
            axs.legend(["Original", "Interpolated"])
            plt.grid()
            plt.tight_layout()
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.savefig(
                f"DAPs_Code/Task3/Graphics/Interpolate/interpolate_{column}.pdf"
            )
            plt.close()

        self.dataframe = self.dataframe.ffill().bfill()

    def outlier_processing(self, threshold: float = 2.5, window: int = 7):
        """
        Function: outlier_processing

        Detects outliers within a set threshold. Outliers are detected through
        their Z score. Outliers are interpolated by calcuating the average value
        of x amount of points around it. It also creates plots containing views
        from before and after outlier processing. Updates dataframe stored within 
        class.

        Args:
            threshold (float): z score that sets outlier cut off point
            window (int): window to interpolate from.

        Returns:
            None

        Example:
            processed_data = DataPreProcessing(dataframe).outlier_processing()
        """
        for column in self.column_name:
            selected_column = self.dataframe[column].copy().reset_index(drop=True)
            _, axs = plt.subplots(figsize=(5, 3))
            axs.plot(self.dataframe.index, self.dataframe[column])
            z_scores = zscore(selected_column)
            outliers = selected_column[abs(z_scores) > threshold]

            if not outliers.empty:
                self.dataframe = self.dataframe.reset_index()

                for i in outliers.index:
                    start_index = max(0, i - window)
                    end_index = min(len(selected_column), i + window + 1)
                    mean_value = selected_column.iloc[start_index:end_index].mean()
                    self.dataframe.at[i, column] = mean_value

                self.dataframe = self.dataframe.set_index("date")

            axs.plot(self.dataframe.index, self.dataframe[column], color="#d62728")
            axs.set_title("Interpolated data for " + str(column))
            axs.legend(["Original", "Clamped"])
            axs.set_xlabel("Date")
            axs.set_ylabel("Value")
            plt.grid()
            plt.tight_layout()
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.savefig(f"DAPs_Code/Task3/Graphics/Outliers/Outliers_{column}.pdf")
            plt.close()

    def data_transformation(self):
        """
        Function: data_transformation

            Performs data transformation and on dataset. This includes
            normalisation and PCA testing.

        Args:
            None
        Returns:
            self.dataframe (pd.Dataframe): dataframe of processed data

        Example:
            processed_data = DataPreProcessing(dataframe).data_transformation()
        """
        self.normalise()
        print("Normalised Data")

        self.pca_test()
        print("Performed PCA Test")

        return self.dataframe.copy()

    def normalise(
        self,
        norm_dict={
            "open": "zscore",
            "high": "zscore",
            "low": "zscore",
            "close": "zscore",
            "volume": "zscore",
        },
    ):
        """
        Function: normalise

        Normalises the dataframe given a desired technque. These include
        min max, log and zscore normalisation. Final result is updated
        within the final database.

        Args:
            norm_dict (dict): dictionary cotaining how different columns should
                                be normalised.

        Returns:
            None

        Example:
            processed_data = DataPreProcessing(dataframe).normalise()
        """
        column_names = self.dataframe.columns
        for column in column_names:
            _, axs = plt.subplots(2, 1, figsize=(5, 3), sharex=True)
            axs[0].plot(self.dataframe.index, self.dataframe[column])

            try:
                tecnique = norm_dict[column]
            except Exception:
                tecnique = "zscore"

            match tecnique:
                case "minmax":
                    self.dataframe[column] = (
                        self.dataframe[column] - self.dataframe[column].min()
                    ) / (self.dataframe[column].max() - self.dataframe[column].min())

                case "zscore":
                    self.dataframe[column] = (
                        self.dataframe[column] - self.dataframe[column].mean()
                    ) / self.dataframe[column].std()

                case "log":
                    self.dataframe[column] = np.log(self.dataframe[column])

                case _:
                    self.dataframe[column] = (
                        self.dataframe[column] - self.dataframe[column].min()
                    ) / (self.dataframe[column].max() - self.dataframe[column].min())

            axs[1].plot(self.dataframe.index, self.dataframe[column], color="#d62728")
            axs[0].set_ylabel("Value")
            axs[1].set_xlabel("Date")
            axs[1].set_ylabel("Norm. Value")
            plt.suptitle(f"{tecnique} Normalisation for column {column}")
            axs[0].grid()
            axs[1].grid()
            plt.tight_layout()
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.savefig(f"DAPs_Code/Task3/Graphics/Normal/{tecnique}_{column}.pdf")
            plt.close()

    def pca_test(self, threshold: float = 0.95):
        """
        Function: pca_test

        Performs PCA testing to describe the variance within the data, and
        drops columns based of results.

        Updates final dataframe stored within class.

        Produces a scree plot stored in graphics.

        Args:
            threshold (float): Value to determin cut off point for PCA testing

        Returns:
            None
        Example:
            processed_data = DataPreProcessing(dataframe).pca_test()
        """

        df_subset = self.dataframe.copy()
        df_subset = pd.concat(
            [df_subset["close"], df_subset.drop(columns=["close"])], axis=1
        )

        scale = StandardScaler()
        stand_div = scale.fit_transform(df_subset)

        pca_instance = PCA()
        pca_result = pca_instance.fit_transform(stand_div)

        # Scree plot
        x_axis = np.linspace(
            1, len(self.dataframe.columns), num=len(self.dataframe.columns)
        )
        plt.figure(figsize=(6, 3))
        plt.plot(
            x_axis, np.cumsum(pca_instance.explained_variance_ratio_), color="#d62728"
        )
        plt.bar(x_axis, pca_instance.explained_variance_ratio_)
        plt.title("PCA - Explained Variance per component")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained variance")
        plt.legend(["Cumulative Sum", "Componenet Variance"])
        plt.tight_layout()

        plt.savefig("DAPs_Code/Task3/Graphics/Scree_plot.pdf")
        plt.close()

        pca_instance = PCA(n_components=2)
        pca_result = pca_instance.fit_transform(stand_div)

        pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
        pca_df["class"] = np.random.choice(["A", "B", "C"], size=len(pca_df))

        # Plotting points with different colors based on the 'class' column
        plt.figure(figsize=(8, 8))
        for class_label, color in zip(
            pca_df["class"].unique(), ["red", "blue", "green"]
        ):
            class_data = pca_df[pca_df["class"] == class_label]
            plt.scatter(
                class_data["PC1"], class_data["PC2"], label=class_label, color=color
            )

        plt.title("Scatter Plot of PC1 against PC2")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig("DAPs_Code/Task3/Graphics/PCA_plot.pdf")
        plt.close()
