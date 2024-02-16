# Stock Forecaster
## Overview
This repository contains a comprehensive stock forecasting system that leverages public APIs, MongoDB for data storage, TensorFlow's LSTM for forecasting, and a Q-learning trading algorithm. The system encompasses data retrieval, processing, analysis, and predictive modeling to assist users in making informed decisions in the stock market.

# Features
Data Retrieval: Utilizes public APIs to fetch real-time stock market data.

Data Storage: Stores fetched data in MongoDB for efficient and organized data management.

WEB API: CRUD Web Api to fetch and interact with Database

Data Processing and Analysis: Performs full-scale data processing and analysis to extract meaningful insights.

TF LSTM Forecasting: Implements a forecasting model using TensorFlow's Long Short-Term Memory (LSTM) neural network for accurate stock price predictions.

Q-learning Trading Algorithm: Includes an implementation of a Q-learning algorithm for automated trading decisions.


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/stock-forecaster.git
    cd stock-forecaster
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate      # On Windows, use 'venv\Scripts\activate'
    ```

3. Install dependencies using `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, you can use `conda` with the provided `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    conda activate stock-forecaster
    ```

4. Configure API keys:
    - Obtain API keys for the data sources used (e.g., Alpha Vantage).
    - Update the configuration file (`config.yaml`) with your API keys.

5. Set up MongoDB:
    - Ensure you have a running instance of MongoDB.
    - Update the MongoDB connection details in the configuration file.

6. Run the data retrieval script:

    ```bash
    python data_retrieval.py
    ```

7. Run the data processing and analysis script:

    ```bash
    python data_processing.py
    ```

8. Train and evaluate the LSTM model:

    ```bash
    python lstm_forecasting.py
    ```

9. Execute the Q-learning trading algorithm:

    ```bash
    python q_learning_trading.py
    ```

## License

This project is licensed under the [MIT License](LICENSE).
