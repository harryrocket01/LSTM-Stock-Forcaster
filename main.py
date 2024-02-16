"""
DAPS Final Assessment - main.py
Forcasting and Predicting Stock
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""

# File Imports
from DAPs_Code.Task1.data_aquisition import DataAquisition
from DAPs_Code.Task2.data_storage import DataStorage
from DAPs_Code.Task3.data_preprocessing import DataPreProcessing
from DAPs_Code.Task4.data_exploration import DataExploration, FinIndicators
from DAPs_Code.Task5.forcasting import Forcasting
from DAPs_Code.Task6.decision_making import DecisionMaking


def main():
    """
    Function: main

    This function runs the final code for the DAPs 2023/24 assignement.

    Each task is broken down through inline comments, with the functions
    used to execute the task set out. This code requires a full wifi
    connection in order to run correctly, as its based around web hosted
    API's and Databases. It was built and developed on windows 10 on a
    intel 10th gen i7 processor.

    Args:
        None

    Returns:
        None

    Example:
            $ python main.py
    """

    print("\n\n ----- Data Acquisition -----")
    # Stock prices over time (Task 1.1)
    data_acquisition = DataAquisition(
        start_date="2019-04-01", end_date="2023-03-31", stock_symbol="AAL"
    )
    stock_data = data_acquisition.retreave_stock()

    # Additional data (Task 1.2)
    weather_data = data_acquisition.retreave_weather()
    trend_data = data_acquisition.retreave_trends()
    oil_data = data_acquisition.retreave_oil()
    cpi_data = data_acquisition.retreve_CPI()

    print("\n\n ----- Data Storage -----")
    # Data storage (Task 2.1)
    DataStorage().database_insert(
        stock_data, data_format="Stock_Data", target_collection="Stock_Data"
    )
    DataStorage().database_insert(
        weather_data, data_format="Weather_Data", target_collection="Weather_Data"
    )
    DataStorage().database_insert(
        oil_data, data_format="Oil_Data", target_collection="Oil_Data"
    )
    DataStorage().database_insert(
        cpi_data, data_format="Finance_Data", target_collection="Finance_Data"
    )
    DataStorage().database_insert(
        trend_data, data_format="Trend_Data", target_collection="Trend_Data"
    )
    db_request = DataStorage().aggregate_request(
        collection="Stock_Data",
        additions=["Finance_Data", "Oil_Data", "Weather_Data", "Trend_Data"],
    )
    # API implementation (Task 2.2)
    """
    Copy of python API using Flask found within the directory ./DAPs_Code/Task2/API.py
    """

    print("\n\n ----- Data Preprocessing -----")
    datapreprocessing_instance = DataPreProcessing(db_request)
    # Data cleaning & visualization (Task 3.1 & 3.2)
    preprocessed_data = datapreprocessing_instance.pre_process()
    # Data transformation & visualization  (Task 3.3 & 3.2)
    transformed_data = datapreprocessing_instance.data_transformation()

    print("\n\n ----- Data Exploration -----")
    # EDA on data (Task 4.1)
    DataExploration(preprocessed_data).derrive_insights()
    # Computing, visualising and deriving insights of known financial indicators (Task 4.2)
    fin_indicators = FinIndicators(preprocessed_data).calculate_indicators()

    print("\n\n ----- Forcasting -----")
    # Development of model using stocks (Task 5.1)

    # Requesting april 2023 data
    result_aquisition = DataAquisition(
        start_date="2023-04-01", end_date="2023-04-30", stock_symbol="AAL"
    )
    result_data = result_aquisition.retreave_stock()
    DataStorage().database_insert(
        result_data,
        data_format="Stock_Data",
        target_collection="Forcast_Results",
    )
    true_result_df = DataStorage().request_all(collection_name="Forcast_Results")

    # Model 1 - only stock movements
    model_1_train = transformed_data[["close"]]
    # Model 2 - stock with auxillary data
    model_2_train = transformed_data[
        ["close", "volume", "value_CPI", "AAL", "temperature_2m_mean"]
    ]
    model1_result = Forcasting().train_forcaster(
        learningrate=0.01,
        epochs=50,
        dropout=0.2,
        window=30,
        forcast_depth=19,
        train=model_1_train,
    )
    # Development of model using stocks and other data sources (Task 5.2)
    model2_result = Forcasting().train_forcaster(
        learningrate=0.01,
        epochs=50,
        dropout=0.2,
        window=30,
        forcast_depth=19,
        train=model_2_train,
    )
    # Implementation of evaluation metrics (Task 5.3)
    Forcasting().prediction_analysis(
        preprocessed_data, true_result_df, model1_result, model2_result
    )

    print("\n\n ----- Descision-Making -----")
    # Implementation of a buy/sell/hold agent (Task 6)
    agent_data = db_request.copy().tail(30)
    agent_instance = DecisionMaking(
        dataframe=agent_data, reward_vector=[0, 0, 0], fin_indicators=fin_indicators
    )
    agent_instance.train(episodes=300, decay=0.98, weight=1)
    agent_instance.buy_sell_reccomendation()


if __name__ == "__main__":
    main()
