"""This modules contains code to be executed after the anonymization kernel has been run"""
import logging

import datetime

import pandas as pd
from anytree import AnyNode
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PostProcessor():
    """The postprocessor will actually recode sensitive terms and make a pretty version of the anonymized dataframe"""

    def __init__(self, config, pp):
        self.__config = config
        self.__preprocessor = pp

    def clean(self, df):
        """
        Takes a dataframe and drops all helper attributes
        Parameters
        ----------
        df: DateFrame
            DataFrame to clean.
        Returns
        -------
        DataFrame
            Cleaned DataFrame.
        """
        df = df.drop(self.__preprocessor.get_non_redundant_entity_attributes(), axis=1)
        df = df.drop(self.__preprocessor.get_redundant_entity_attributes(), axis=1)
        return df

    def uncompress(self, df):
        """
        Takes a dataframe and uncompresses it using the first textual attribute available
        Parameters
        ----------
        df: DateFrame
            DataFrame to uncompress.
        Returns
        -------
        DataFrame
            Uncompressed DataFrame.
        """
        column_to_uncompress = None
        if len(self.__config.get_textual_attributes()) > 0:
            column_to_uncompress = self.__config.get_textual_attributes()[0]  # Take first column to uncompress

        logger.info("Uncompressing dataframe on attribute %s", column_to_uncompress)

        if column_to_uncompress:
            uncompressed_df = pd.DataFrame(columns=df.columns)
            for index in tqdm(range(len(df)), total=len(df), desc="Uncompressing"):
                if isinstance(df.loc[index, column_to_uncompress], list):
                    insensitive_attributes = self.__config.get_insensitive_attributes()
                    textual_attributes = self.__config.get_textual_attributes()
                    to_drop = textual_attributes + insensitive_attributes
                    raw_row = df.drop(to_drop, axis=1).loc[index]
                    for ii in range(len(df.loc[index, column_to_uncompress])):
                        row = raw_row
                        for insensitive_attribute in insensitive_attributes:
                            value_to_append = df.loc[index, insensitive_attribute][ii]
                            row = row.append(pd.Series([value_to_append], index=[insensitive_attribute]))

                        for textual_attribute in textual_attributes:
                            text_to_append = df.loc[index, textual_attribute][ii]
                            row = row.append(pd.Series([text_to_append], index=[textual_attribute]))
                        uncompressed_df = uncompressed_df.append(row, ignore_index=True)
                else:
                    data_row = df.loc[index]
                    uncompressed_df = uncompressed_df.append(data_row, ignore_index=True)
        return uncompressed_df

    def pretty(self, df):
        """
        Takes a dataframe and makes values in columns pretty
        Parameters
        ----------
        df: DateFrame
            DataFrame to uncompress.
        Returns
        -------
        DataFrame
            Prettyfied DataFrame.
        """
        pretty_df = pd.DataFrame(columns=df.columns)
        logger.info("Converting values in anonymized dataframe to their pretty versions")
        for col in df.columns:
            for index, value in tqdm(df[col].iteritems(), total=len(df), desc=col):
                pretty_df.at[index, col] = convert_to_pretty(value, self.__config.get_default_date_format())
        return pretty_df


def convert_to_pretty(value, date_format="%Y-%m-%d"):
    """
        Takes any value and transforms it to a pretty version
        Parameters
        ----------
        value: any
            Value to make pretty.
        date_format: str
            Date format to use for date values.
        Returns
        -------
        any
            Prettyfied value.
        """
    if isinstance(value, AnyNode):
        return convert_to_pretty(value.range)
    elif isinstance(value, range):
        return "[{}-{}]".format(value.start, value.stop - 1)
    elif isinstance(value, (pd.Timestamp, datetime.datetime)):
        return value.strftime(date_format)
    elif isinstance(value, pd.Period):
        return str(value)
    elif isinstance(value, (set, frozenset)):
        sorted_values = sorted(list(value), key=str.lower)
        res = ','.join(str(e) for e in sorted_values)
        return "({})".format(res)
    elif isinstance(value, list):
        sorted_values = sorted(value, str.lower)
        res = ','.join(str(e) for e in sorted_values)
        return "[{}]".format(res)
    elif pd.isnull(value):
        return ''
    else:
        return str(value)
