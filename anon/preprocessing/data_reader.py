"""This module includes code to read and format csv data accordingly to a config"""
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype


class DataReader:
    """This class is a wrapper for reading a input data"""

    def __init__(self, config):
        self.__config = config

    def read(self, input_file):
        """
        Reading given input file and returning a dataframe
        Parameters
        ----------
        input_file: (str, Path)
            Input file path.
        Returns
        -------
        DataFrame
            Data read in DataFrame.
        """
        date_attributes = self.__config.get_date_attributes()
        date_formats = self.__config.get_date_formats()
        textual_attributes = self.__config.get_textual_attributes()
        data_types = self.__config.get_data_types()
        ordinal_orders = self.__config.get_ordinal_orders()

        df = pd.read_csv(input_file)
        df.columns = map(str.lower, df.columns)
        df = df.dropna(subset=df.columns.difference(textual_attributes), axis='index', how='any')  # drop any rows with empty values except for the textual ones

        for date_attribute in date_attributes:
            if not is_datetime64_any_dtype(df[date_attribute]):
                df[date_attribute] = pd.to_datetime(df[date_attribute], format=date_formats[date_attribute], errors='coerce')
            df = df.dropna(subset=[date_attribute])  # drop rows if there are empty timestamps

        df = df.dropna(subset=df.columns.difference(textual_attributes), axis='index', how='any')  # drop any rows with empty values except for the textual ones

        df = df.astype(data_types)

        for ordinal_attribute in ordinal_orders:
            df[ordinal_attribute].cat.reorder_categories(ordinal_orders[ordinal_attribute], ordered=True, inplace=True)
        return df
