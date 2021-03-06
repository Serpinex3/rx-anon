"""This module contains utility functions required within the anonymization kernel"""
import pandas as pd
import numpy as np
from anytree import AnyNode
from nlp.similarity_module import compare_complete_match
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_categorical_dtype


def reduce_string(element):
    """Takes an string which should be generalized and returns a more general version (one step)"""
    if len(element) == 1:
        return element
    chars = list(element)
    if chars.count('*') > 0:
        chars[len(chars) - 2] = '*'
        result = ''.join(chars)
        result = result[:len(chars) - 1]
    else:
        chars[len(chars) - 1] = '*'
        result = ''.join(chars)
    return result


def next_string_to_reduce(strings):
    """Provides the next element to reduce given a list of strings"""
    length = 0
    for element in strings:
        if len(element) > length:
            result = element
            length = len(element)
        elif len(element) == len(result) and '*' not in element:
            result = element
    return result


def intersect_token_lists(list_1, list_2):
    """Takes two lists with tokens and returns two sets where the first contains a the intersection of both and the second contains unique elements"""
    intersection = set()

    for span_1 in list_1:
        for span_2 in list_2:
            result = compare_complete_match(span_1, span_2)
            if result:
                intersection.add(span_1)
                intersection.add(span_2)
    unique = set(list_1).intersection(intersection)
    return intersection, unique


def must_be_flattened(series):
    """Checks whether a series must be flattened. This is the case if it contains multiple values for one entry"""
    for element in series:
        if isinstance(element, frozenset):
            return True
    return False


def is_token_list(series):
    """Checks whether a series contains a list of tokens"""
    all_none = True
    for element in series:
        if isinstance(element, list):
            return True
        if element:
            all_none = False
    return all_none


def is_node(series):
    """Checks whether series contains any nodes"""
    for element in series:
        if isinstance(element, AnyNode):
            return True
    return False


def flatten_set_valued_series(series):
    """Takes a series and flattens it by resolving sets within the series"""
    flattened = list()
    indexes = list()
    is_category = False
    for index, element in series.iteritems():
        if isinstance(element, frozenset):
            for item in element:
                flattened.append(item)
                indexes.append(index)
        elif isinstance(element, list):  # List of terms
            for item in element:
                flattened.append(item.text.lower())
                indexes.append(index)
            is_category = True
        elif element is None:
            flattened.append(element)
            indexes.append(index)
            is_category = True
        else:
            flattened.append(element)
            indexes.append(index)
    return flattened, indexes, is_category


def agg_mean(series):
    """Aggregate series values by calculating the mean"""
    if is_numeric_dtype(series):
        return series.mean()
    elif is_datetime64_any_dtype(series):
        return pd.to_datetime(series.dropna().astype(np.int64).mean())
    else:
        raise Exception("Could not aggregate since no option for mean")


def agg_categorical(series):
    """Aggregate categorical series by returning the mode"""
    if not is_categorical_dtype(series):
        raise Exception("Could not aggregate since no categorical")

    if series.isnull().all():
        return None

    mode = series.mode().unique()[0]
    return mode
