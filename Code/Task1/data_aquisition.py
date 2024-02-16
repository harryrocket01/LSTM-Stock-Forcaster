"""
DataAquision.py
Forcasting and Predicting Stock
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""

# Python Module Imports
import requests
import pandas as pd
from pytrends.request import TrendReq


class DataAquisition:
    """
    Class: DataAquisition

    This class handles the requests of stock and extra data through API
    requests and Webscraping

    Each of the function returns its relevent data, error handling where
    approprate.

    Attributes:
        start_date (str): start date to collect data from
        end_date (str): end date to collect data to
        stock_symbol (str): stock symbol of stock to retreave

    Methods:
        retreave_stock () : Retreaves stock Data
        retreave_weather () : Retreaves weather Data
        retreave_CPI () : Retreaves cpi Data
        retreave_oil () : Retreaves oil Data
        retreave_trends () : Retreaves trends Data

    Args:
        start_date (str): start date to collect data from
        end_date (str): end date to collect data to
        stock_symbol (str): stock symbol of stock to retreave

    Example:
            DataAquisition().retreavestock()
    """

    def __init__(
        self,
        start_date: str = "2019-04-01",
        end_date: str = "2023-03-31",
        stock_symbol: str = "AAL",
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.stock_symbol = stock_symbol

    def retreave_stock(self):
        """
        Function: retreave_stock

            Retreaves stock movement data through alpha vantage API
            request. Formats the data and returns as a Dataframe.

        Args:
            None

        Returns:
            retreaved_data (pd.Dataframe): Dataframe of retreaved data

        """
        parameters = {
            "function": "TIME_SERIES_DAILY",
            "symbol": self.stock_symbol,
            "datatype": "json",
            "outputsize": "full",
        }

        finace_instance = FinanceData()
        retreaved_data = finace_instance.request_av(
            "Stock Data", parameters, start_date=self.start_date, end_date=self.end_date
        )

        return retreaved_data

    def retreave_weather(self):
        """
        Function: retreave_weather

            Retreaves weather data from open metro Resful Api.
            Data is collected to dataframe.

        Args:
            None

        Returns:
            retreaved_data (pd.Dataframe): Dataframe of retreaved data

        """
        weather_instance = WeatherData()
        retreaved_data = weather_instance.request(
            start_date=self.start_date, end_date=self.end_date
        )

        return retreaved_data

    def retreve_CPI(self):
        """
        Function: retreve_CPI

        Retreaves CPI movements from alpha vantage for final
        task.

        Args:
            None

        Returns:
            retreaved_data (pd.Dataframe): Dataframe of retreaved data

        """
        parameters = {"function": "CPI", "interval": "monthly", "datatype": "json"}

        finace_instance = FinanceData()
        retreaved_data = finace_instance.request_av(
            "CPI Data", parameters, start_date=self.start_date, end_date=self.end_date
        )

        return retreaved_data

    def retreave_oil(self):
        """
        Function: retreave_oil

            Retreaves oil price data from alpha vantage for
            wti and brent, for final taks.

        Args:
            None

        Returns:
            merged_data (pd.Dataframe): Dataframe of retreaved data

        """
        parameters_wti = {"function": "WTI", "interval": "daily", "datatype": "json"}
        finace_instance = FinanceData()
        wti_data = finace_instance.request_av(
            "OIL Data",
            parameters_wti,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        parameters_brent = {
            "function": "BRENT",
            "interval": "daily",
            "datatype": "json",
        }
        brent_data = finace_instance.request_av(
            "OIL Data",
            parameters_brent,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        try:
            if isinstance(wti_data, pd.DataFrame) and isinstance(
                brent_data, pd.DataFrame
            ):
                merged_data = pd.merge(
                    wti_data,
                    brent_data,
                    left_index=True,
                    right_index=True,
                    suffixes=("_WTI", "_Brent"),
                )
            else:
                merged_data = None
        except Exception:
            merged_data = None

        return merged_data

    def retreave_trends(self):
        """
        Function: retreave_trends

            Retreaves trend data from google trends for
            chosen key words for final taks.

        Args:
            None

        Returns:
            merged_data (pd.Dataframe): Dataframe of retreaved data

        """
        keywords = ["AAL", "Covid", "Flights", "holiday"]

        trend_instance = TrendData()
        retreaved_data = trend_instance.request(
            keywords=keywords, start_date=self.start_date, end_date=self.end_date
        )
        return retreaved_data


class FinanceData:
    """
    Class: FinanceData

    Collects stock data throuhg the use of the Alpha vantage API.
    Once collected, json is formatted to dataframe, and filtered
    to desired time frame. Dataframe is returned.

    Code based off of Labs 1/2 and code Doc:
    https://www.alphavantage.co/documentation/

    Attributes:
        stock_data_url (str): API url requested through
        api_key (str): API request key
        Data (pd.Dataframe): Dataframe of retreaved data
        end_date (str): end date to collect data to
        stock_symbol (str): stock symbol of stock to retreave

    Methods:
        retreave_av () : performs alpha vantage request

    Args:
        None

    Example:
            DataAquisition().retreavestock()
    """

    def __init__(self):
        self.stock_data_url = "https://www.alphavantage.co/query"
        self.api_key = "APIKEY"

        self.data = None

        self.start_date = "2019-04-01"
        self.end_date = "2023-03-31"

    def request_av(
        self,
        data_type: str,
        parameters: dict,
        start_date: str = "2019-04-01",
        end_date: str = "2023-03-31",
    ):
        """
        Function: request_av

        Collects data through the use of the Alpha vantage API.

        Once collected, json is formatted to dataframe, and filtered
        to desired time frame. Dataframe is returned.

        Args:
            data_type - str - handles the type of request, so it can
            format the recived json correctly
            paramerters - dict - dictionary of request parameters,
            based off of the AlphaVantage API Documentation
            start_date - string - date to start the request, in the
            form of %Y %m %d
            end_date - string - date to end the request, in the
            form of %Y %m %d

        Returns:
            time_filtered_data - Pandas Dataframe - On a successful
            request, the requested data will be returned. Otherwise
            None is retured in the event of a unsucessful request

        Example:
            FinanceData().request_av()
        """

        try:
            parameters["apikey"] = self.api_key
            self.start_date = start_date
            self.end_date = end_date
        except Exception:
            print("Peramters for request are incorrect")
            return None

        response = requests.get(self.stock_data_url, params=parameters, timeout=30)

        # Successful Request
        if response.status_code == 200:
            response_json = response.json()
            # Checks if rate limit has been reached
            if "Information" in response_json:
                if (
                    response_json["Information"]
                    == "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day. Please subscribe to any of the premium plans at https://www.alphavantage.co/premium/ to instantly remove all daily rate limits."
                ):
                    print("\nFailed to Reach API - Alpha Vantage - Free Limit Reached")
                    return None

            # Extract data from JSON
            if data_type == "Stock Data":
                data = response.json()["Time Series (Daily)"]
                unfilterd_data = pd.DataFrame(data).T
                time_filtered_data = unfilterd_data[
                    (unfilterd_data.index >= self.start_date)
                    & (unfilterd_data.index <= self.end_date)
                ]
            else:
                data = response_json["data"]
                unfilterd_data = pd.DataFrame(data)
                unfilterd_data = unfilterd_data.set_index("date")

            # Filter the constrained dates
            time_filtered_data = unfilterd_data[
                (unfilterd_data.index >= self.start_date)
                & (unfilterd_data.index <= self.end_date)
            ]
            self.data = time_filtered_data

            print(f"\nRetreaved {data_type} From Alpha Vantage")
            return time_filtered_data

        else:
            print(
                f"\nFailed to Reach API - Alpha Vantage - Error Code:{response.status_code}"
            )
            return None


class WeatherData:
    """
    Class: WeatherData

    Collects weather data throuhg the use of the OpenMetro API.


    Code based off of Labs 2/3/4 and doc:
    https://open-meteo.com/

    Attributes:
        stock_data_url (str): API url requested through
        hub_locations (Array): Array of coordinates where American airlines
         are

    Methods:
        request () : performs open metro api request

    Args:
        None
    """

    def __init__(self):
        self.url = "https://archive-api.open-meteo.com/v1/archive"

        # American Airline operate 10 hubs, these are the co-ordiantes
        self.hub_locations = {
            "CLT": ["35.2271", "80.8431"],
            "ORD": ["41.9802", "87.9090"],
            "DFW": ["32.7079", "96.9209"],
            "LAX ": ["33.9438", "118.4091"],
            "MIA": ["25.7951", "80.2795"],
            "JFK": ["40.6446", "73.7797"],
            "LGA": ["40.7769", "73.8740"],
            "PHL": ["39.8731", "75.2437"],
            "PHX": ["33.4352", "112.0101"],
            "DCA": ["38.8512", "77.0402"],
        }

    def request(self, start_date: str = "2019-04-01", end_date: str = "2023-03-31"):
        """
        Function: request

        Collects stock data throuhg the use of the OpenMetro API. All
        data is placed it in a dataframe and returned.

        Args:
            start_date - string - date to start the request, in the
            form of %Y %m %d
            end_date - string - date to end the request, in the
            form of %Y %m %d

        Returns:
            merged_data - Pandas Dataframe - On a successful request,
            the requested data will be returned. Otherwise None is
            retured in the event of a unsucessful request

        Example:
            WeatherData().request()
        """
        data = []

        # itrate through each of the hub locations
        for hub in self.hub_locations:
            params = {
                "latitude": self.hub_locations[hub][0],
                "longitude": self.hub_locations[hub][1],
                "start_date": start_date,
                "end_date": end_date,
                "daily": [
                    "weather_code",
                    "temperature_2m_mean",
                    "precipitation_sum",
                    "rain_sum",
                    "snowfall_sum",
                    "wind_speed_10m_max",
                ],
            }

            response = requests.get(self.url, params=params, timeout=30)

            # On Successful Request
            if response.status_code == 200:
                retreaved_data = response.json()
                data_df = pd.DataFrame(retreaved_data)
                data_df = data_df["daily"].T

                etracted = {
                    "date": data_df["time"],
                    f"weather_code_{hub}": data_df["weather_code"],
                    f"temperature_2m_mean_{hub}": data_df["temperature_2m_mean"],
                    f"precipitation_sum_{hub}": data_df["precipitation_sum"],
                    f"rain_sum_{hub}": data_df["rain_sum"],
                    f"snowfall_sum_{hub}": data_df["snowfall_sum"],
                    f"wind_speed_10m_max_{hub}": data_df["wind_speed_10m_max"],
                }

                formatted_df = pd.DataFrame(etracted)
                formatted_df = formatted_df.set_index("date")
                data.append(formatted_df)

            else:
                print(
                    f"\nFailed to Reach API - Open Metro - Error Code:{response.status_code}"
                )
                return None

        # Concantinates and Returns all of the collected Data
        merged_data = pd.concat(data, axis=1)
        print("\nRetreaved Weather From Open Metro")
        return merged_data


class TrendData:
    """
    Class: TrendData

    Collects trend data through the use of pytrends package. Code
    based off pytrends documenation. Data is recived as a dataframe
    hence it drops all unessasarry columns and passes back the
    concantinated dataframe.

    Please not, pytrends is a webscrabing function that relys off
    of access to google. Please dont run it on a pc with a VPN.

    It also blocks MAC and IP address based off too much use.

    Attributes:
        pytrends (): Pytrends request instance

    Methods:
        request () : performs pytrend request

    Args:
        None

    """

    def __init__(self):
        self.pytrends = TrendReq()

    def request(
        self,
        keywords: list = ["holidays"],
        start_date: str = "2019-04-01",
        end_date: str = "2023-03-31",
    ):
        """
        Function: request

        Collects trend data throuhg the use of the pytrends. Formats
        the data to a dataframe and returns it.

        Args:
            keywords - array - array of words to be collected through
            google trends.
            start_date - string - date to start the request, in the
            form of %Y %m %d
            end_date - string - date to end the request, in the
            form of %Y %m %d

        Returns:
            merged_data - Pandas Dataframe - On a successful request,
            the requested data will be returned. Otherwise None is
            retured in the event of a unsucessful request

        Example:
            TrendData().request()
        """
        to_return = []

        date = f"{start_date} {end_date}"

        # Iterates through passed keywords
        for word in keywords:
            try:
                self.pytrends.build_payload(
                    [word],
                    geo="US",
                    timeframe=date,
                )
                reply = self.pytrends.interest_over_time()
                reply = reply.drop(["isPartial"], axis=1)

                to_return.append(reply)
            except Exception:
                pass

        if to_return:
            print("\nRetreaved data From Google Trends")
            merged_data = pd.concat(to_return, axis=1)
            return merged_data

        print("\nFailed to Reach API - Google Trends")
        return None
