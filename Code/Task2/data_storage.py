"""
DAPS Final Assessment - DataStorage.py
Forcasting and Predicting Stock
Author - Harry Softley-Graham 
Written - Nov 2023 - Jan 2024
"""

# Python Package Imports
from datetime import timedelta
from typing import Any
from pymongo import MongoClient
import pandas as pd


class DataStorage:
    """
    Class: DataStorage

    This class handles the storage of the passed data, to store within a
    pymongo NoSQL database.

    Code based off Labs 1/2 and work done in the hackathon.

    Atributes:
        url (str): MongoDB endpoint URL
        client (): MongoDB Connection
        database (): DAPs_Assessment Database Endpoint
        collections (): Collections to be created within the database

    Methods:
        create_database() : creates database
        create_collections() : creates collections
        database_insert() : inserts value in to target database
        stock_format() : formats stock data
        oil_format() : formats oil data
        AV_format() : formats alpha vantage request
        trend_format() : formats trends data
        weather_format() : formats weather data
        aggregate_request() : requests using aggregate
        request_all() : requests all data

    Example:
            instance = DataStorage()
    """

    def __init__(self):
        url = "MONGODB URL"

        self.client = MongoClient(url)
        self.create_database()
        self.database = self.client["DAPs_Assessment"]

        collections = [
            "Stock_Data",
            "Finance_Data",
            "Trend_Data",
            "Oil_Data",
            "Weather_Data",
            "Forcast_Results",
        ]

        self.create_collections(collections)

    def create_database(self, name: str = "DAPs_Assessment"):
        """
        Function: create_database

        Sets up a databse within pymongom if it already has not
        been created.

        Args:
            name - string - name of target databse
        """
        self.client[name]

    def create_collections(self, collections: list):
        """
        Function: create_database

        Creates desired collections within target database in the,
        case it has not been made already

        Args:
            collections - Array - array of names of collections
        """

        current_collections = self.database.list_collection_names()

        for name in collections:
            if name not in current_collections:
                self.database.create_collection(name)

    def database_insert(self, data: Any, data_format: str, target_collection: str):
        """
        Function: database_insert

        Collects data through the use of the Alpha vantage API

        Args:
            data - Any - data to be store upon database
            data_format - string - choice of data format, typically the
            same as target_collection
            target_collection - string - target collection that the
            data should be stored at.

        Returns:
            result - dict - dictonary containg data if a action is
            successful. If unsucessful None is returned.

        Example:
            DataStorage().database_insert(data, format, target_collection)
        """

        # Check passed data exists
        try:
            if data in (None, []):
                print("Inputted Data Empty")
                return None
        except Exception:
            pass

        stock_collection = self.database[target_collection]
        formatted_data = []

        # calls correct data format
        match data_format:
            case "Stock_Data":
                formatted_data = self.stock_format(data)
            case "Oil_Data":
                formatted_data = self.oil_format(data)
            case "Finance_Data":
                formatted_data = self.av_format(data)
            case "Weather_Data":
                formatted_data = self.weather_format(data)
            case "Trend_Data":
                formatted_data = self.trend_format(data)
            case _:
                print("Target Collection Does Not Exist")
                return None

        # Checks exisiting database for duplicates
        existing_dates = stock_collection.distinct("date")
        new_data = [
            item for item in formatted_data if item["date"] not in existing_dates
        ]
        existing_entries = [
            item for item in formatted_data if item["date"] in existing_dates
        ]

        filter_list = []
        for data_line in existing_entries:
            filter_list.append({"date": data_line["date"]})

        result = None
        # Inserts New Data
        if new_data:
            result = stock_collection.insert_many(new_data)
            print(f"Uploaded New Entries to DB - {target_collection}")

        # Updates exisisting Entries
        if existing_entries and new_data:
            filter_criteria = {
                "date": {"$in": [line["date"] for line in existing_entries]}
            }
            update_data = [{"$set": line} for line in existing_entries]

            # Splits the update_data in to batch of 50 size, as the rate limit is 50
            update_batches = []

            for i in range(0, len(update_data), 50):
                batch = update_data[i : i + 50]
                update_batches.append(batch)

            for batch in update_batches:
                result = stock_collection.update_many(
                    filter_criteria, batch, upsert=True
                )

            print(f"Updated Entries to DB - {target_collection}")

        # Prints and Returns
        if new_data or existing_entries:
            print(f"Uploaded and Updated New Entires to DB - {target_collection}")
            try:
                return result
            except Exception:
                pass
        else:
            print(f"No Data to Append - {target_collection}")

        return None

    def stock_format(self, data: Any) -> list:
        """
        Function: stock_format

        Formats data from Dataframe to array of dictionaries

        Args:
            data - Any - data to be formatted

        Returns:
            formatted_data - array[dict] - returns an array of
            formatted dictionaries, ready to be passed to mongoDB
            add and update fucntions.

        Example:
            formatted_data = DataStorage().stock_format(data)
        """

        formatted_data = []
        for index, row in data.iterrows():
            stock_data = {
                "date": index,
                "open": float(row["1. open"]),
                "high": float(row["2. high"]),
                "low": float(row["3. low"]),
                "close": float(row["4. close"]),
                "volume": float(row["5. volume"]),
            }
            formatted_data.append(stock_data)

        return formatted_data

    def oil_format(self, data: Any) -> list:
        """
        Function: oil_format

        Formats data from Dataframe to array of dictionaries

        Args:
            data - Any - data to be formatted

        Returns:
            formatted_data - array[dict] - returns an array of
            formatted dictionaries, ready to be passed to mongoDB
            add and update fucntions.

        Example:
            formatted_data = DataStorage().oil_format(data)
        """
        formatted_data = []
        for index, row in data.iterrows():
            try:
                value_wti = float(row["value_WTI"])
            except Exception:
                value_wti = float("nan")

            try:
                value_brent = float(row["value_Brent"])
            except Exception:
                value_brent = float("nan")

            data_row = {
                "date": str(index),
                "value_WTI": value_wti,
                "value_Brent": value_brent,
            }
            formatted_data.append(data_row)

        return formatted_data

    def av_format(self, data: Any) -> list:
        """
        Function: av_format

        Formats data from Dataframe to array of dictionaries

        Args:
            data - Any - data to be formatted

        Returns:
            formatted_data - array[dict] - returns an array of
            formatted dictionaries, ready to be passed to mongoDB
            add and update fucntions.

        Example:
            formatted_data = DataStorage().AV_format(data)
        """

        formatted_data = []

        for index, row in data.iterrows():
            try:
                value = float(row["value"])
            except Exception:
                value = float("nan")

            data_row = {"date": str(index), "value_CPI": value}

            formatted_data.append(data_row)

        return formatted_data

    def trend_format(self, data: Any) -> list:
        """
        Function: trend_format

        Formats data from Dataframe to array of dictionaries

        Args:
            data - Any - data to be formatted

        Returns:
            formatted_data - array[dict] - returns an array of
            formatted dictionaries, ready to be passed to mongoDB
            add and update fucntions.

        Example:
            formatted_data = DataStorage().trend_format(data)
        """
        formatted_data = []
        key_words = data.columns.tolist()
        for index, row in data.iterrows():
            date = index + timedelta(days=1)
            data_row = {"date": date.strftime("%Y-%m-%d")}
            for word in key_words:
                data_row[word] = float(row[word])
            formatted_data.append(data_row)

        return formatted_data

    def weather_format(self, data: Any) -> list:
        """
        Function: weather_format

        Formats data from Dataframe to array of dictionaries

        Args:
            data - Any - data to be formatted

        Returns:
            formatted_data - array[dict] - returns an array of
            formatted dictionaries, ready to be passed to mongoDB
            add and update fucntions.

        Example:
            formatted_data = DataStorage().weather_format(data)
        """

        formatted_data = []
        column_titles = data.columns
        for index, row in data.iterrows():
            data_row = {"date": index}

            for column in column_titles:
                data_row[column] = row[column]

            formatted_data.append(data_row)

        return formatted_data

    def aggregate_request(self, collection: str, additions: list):
        """
        Function: aggregate_request

        Performs an aggregate request to mongo DB given
        a target collection.

        Aggretgate request made through aggragte builder,
        and modified to be iterative.

        Args:
            collection - Any - target collection that will be the
            root of the aggregate_request
            additions - Array - array of collections that will provide
            the additional data.

        Returns:
            base_df - Pandas Dataframe - Dataframe of aggregated data from
            mongoDB

        Example:
            formatted_data = DataStorage().aggregate_request(collection, additions)
        """
        # Array of the various dicts to perform aggragate request
        pipeline = []

        # Generate Aggragate Request
        for current in additions:
            pipeline.append(
                {
                    "$lookup": {
                        "from": current,
                        "let": {"date": "$date"},
                        "pipeline": [
                            {"$match": {"$expr": {"$eq": ["$date", "$$date"]}}}
                        ],
                        "as": current.lower(),
                    }
                }
            )
        pipeline.append(
            {
                "$project": {
                    "_id": 0,
                    "date": 1,
                    "Stock_Data": "$$ROOT",
                    **{
                        Current: {"$arrayElemAt": [f"${Current.lower()}", 0]}
                        for Current in additions
                    },
                }
            }
        )

        # Perform request and format to Dataframe
        result = self.database[collection].aggregate(pipeline)

        data = list(result)
        dataframe = pd.DataFrame(data)

        base_df = pd.json_normalize(dataframe[collection]).drop("_id", axis=1)
        for current in additions:
            base_df = base_df.drop(current.lower(), axis=1)

        for current in additions:
            try:
                current_df = pd.json_normalize(dataframe[current]).drop("_id", axis=1)
                base_df = pd.merge(base_df, current_df, on="date", how="left")
            except KeyError:
                pass

        base_df = base_df.sort_values(by="date", ascending=True)

        return base_df

    def request_all(self, collection_name: str = "Stock_Data"):
        """
        Function: request_all

        The request all function pulls everything from the target
        collection. If empty retruns nothing. Passes data back
        as a draframe with date as the index.

        Args:
            collection_name - string - name of target collection

        Returns:
            formatted_data - array[dict] - returns an array of
            formatted dictionaries, ready to be passed to mongoDB
            add and update fucntions.

        Example:
            formatted_data = DataStorage().weather_format(data)
        """
        try:
            collection = self.database[collection_name]
            returned_data = collection.find()
            returned_data_pd = pd.DataFrame(returned_data)
            returned_data_pd = returned_data_pd.set_index("date").drop(["_id"], axis=1)
            print(f"\nData from {collection_name} fetched successfully")
            return returned_data_pd
        except Exception:
            print(f"\nFailed to Fetch {collection_name}")
            return None
