"""
Api.py
Forcasting and Predicting Stock
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""

from typing import Any
from numpy import empty
from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson.json_util import dumps

app = Flask(__name__)


class Api:
    """
    Class: API

    A simple API to allow Melanie to contribute and analyse the data I collected.
    This API follows the CRUD protocol. Code is based off examples in Flask
    documentation, along with labs 1/2/3.

    https://flask.palletsprojects.com/en/3.0.x/

    Attributes:
        self.client (MongoClient): MongoDB Client isntance
        self.mdb_database (MongoDatabase): MongoDB database instance
    Methods:
        create(): create new entry within database
        read(): read from selected area of database
        update(): update value in database
        delete(): delete item in database

    Args:
        db_url (string): database url string to connect to

    Example:
        api = API(dataframe)
    """

    def __init__(self, db_url: str):
        self.client = MongoClient(db_url)
        self.mdb_database = self.client["DAPs_HSG"]

    def create(self, collection: str, item: Any) -> bool:
        """
        Function: create

        Creates new entry at set collection. C of CRUD protocal.

        Args:
            collection (str): Name of collection
            item (Any): item to insert
        Returns:
            True/False - based on success
        Example:
            explore = API(dataframe).create()
        """
        if collection not in self.mdb_database.list_collection_names():
            return False

        try:
            set_collection = self.mdb_database[collection]
            existing_dates = set_collection.distinct("date")

            if item["date"] in existing_dates:
                return False

            for term in item:
                try:
                    item[term] = float(item[term])
                except Exception:
                    pass
            set_collection.insert_one(item)
            return True
        except Exception:
            return False

    def read(self, collection, query: dict) -> Any:
        """
        Function: read

        Reads given values based on search criteria. R of CRUD protocal.

        Args:
            collection (str): Name of collection
            item (Any): reading critiera
        Returns:
            response (json) - Response of items read
        Example:
            explore = API(dataframe).read()
        """
        if collection not in self.mdb_database.list_collection_names():
            return False
        if query is None or query is empty:
            query = {}
        else:
            for request_var in query:
                try:
                    query[request_var] = float(query[request_var])
                except Exception:
                    pass

        set_collection = self.mdb_database[collection]
        response = set_collection.find(query)
        response = dumps(response)
        return response

    def update(self, collection, item: Any, properties: dict) -> bool:
        """
        Function: update

        updates entrys value within a collection. U of CRUD protocal.

        Args:
            collection (str): Name of collection
            item (Any): item to insert
            properties (dict): items to update
        Returns:
            True/False - based on success
        Example:
            explore = API(dataframe).update()
        """
        if collection not in self.mdb_database.list_collection_names():
            return False
        set_collection = self.mdb_database[collection]

        for request_var in properties:
            try:
                properties[request_var] = float(properties[request_var])
            except Exception:
                pass

        result = set_collection.update_one({"date": item}, {"$set": properties})

        return bool(result.modified_count)

    def delete(self, collection, id: str) -> bool:
        """
        Function: create

        Deletes entry based off of its date. D of CRUD protocal.

        Args:
            collection (str): Name of collection
            id (Any): Date to delete
        Returns:
            True/False - based on success
        Example:
            explore = API(dataframe).delete()
        """
        if collection not in self.mdb_database.list_collection_names():
            return False
        set_collection = self.mdb_database[collection]
        result = set_collection.delete_one({"date": id})

        return bool(result.deleted_count)


api = Api(
    "MONGODB URL"
)


@app.route("/create/<string:database>")
def create(database):
    """
    Flask Function: Create
    """
    item = request.args.to_dict()
    print(item)
    success = api.create(database, item)
    return jsonify({"success": success})


# Flask Read Function
@app.route("/read/<string:database>")
def read(database):
    """
    Flask Function: Read
    """
    query = request.args.to_dict()
    response = api.read(database, query)
    return jsonify(response)


# Flask Update Function
@app.route("/update/<string:database>/<string:id>")
def update(database, id):
    """
    Flask Function: Update
    """
    query = request.args.to_dict()
    success = api.update(database, id, query)
    return jsonify({"success": success})


# Flask Delete Function
@app.route("/delete/<string:database>/<string:id>")
def delete(database, id):
    """
    Flask Function: Delete
    """
    response = api.delete(database, id)
    return jsonify({"success": response})


if __name__ == "__main__":
    app.run(debug=True)


# http://127.0.0.1:5000/read/Oil_Data?date=2023-01-17
# http://127.0.0.1:5000/update/Oil_Data/2023-01-17?value=110
# http://127.0.0.1:5000/read/Oil_Data?date=2023-01-17
# http://127.0.0.1:5000/delete/Oil_Data/2023-01-17
# http://127.0.0.1:5000/read/Oil_Data?date=2023-01-17
# http://127.0.0.1:5000/create/Oil_Data?date=2023-01-17&value=100
