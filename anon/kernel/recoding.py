"""This module contains code to recode values to achieve k-anonymity"""
import math
import pandas as pd

from pandas.api.types import (is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype)

from kernel.util import is_token_list, must_be_flattened, flatten_set_valued_series, next_string_to_reduce, reduce_string, intersect_token_lists


def recode(series, recoding_rules=None, hierarchies=None):
    """
    Takes a series and applies appropriate generalization function. Returns a generalized series
    Parameters
    ----------
    series: Series
        Series to be recoded.
    recoding_rules: dict
        Dictionary containing recoding rules.
    hierarchies: dict
        Dictionary containing generalization hierarchies.
    Returns
    -------
    Series
        Recoded series.
    """
    generalization_function = None
    set_valued = False
    if is_token_list(series):
        generalization_function = recode_tokens
    elif must_be_flattened(series):
        generalization_function = recode_set_valued
        set_valued = True
    elif is_numeric_dtype(series):
        generalization_function = recode_range
    elif is_datetime64_any_dtype(series):
        generalization_function = recode_dates
    elif is_categorical_dtype(series):
        if recoding_rules and series.name in recoding_rules and recoding_rules[series.name] == "string_reduction":
            generalization_function = recode_strings
        elif series.cat.ordered:
            generalization_function = recode_ordinal
        else:
            generalization_function = recode_nominal
    else:
        generalization_function = recode_set_valued
        set_valued = True

    if hierarchies and series.name in recoding_rules and recoding_rules[series.name] == "hierarchy" and series.name in hierarchies:
        hierarchy = hierarchies[series.name]
        result = generalization_function(series, hierarchy)
    elif set_valued:
        result = generalization_function(series, recoding_rules, hierarchies)
    else:
        result = generalization_function(series)
    series = series.map(lambda x: result)
    return series


def recode_set_valued(series, recoding_rules, hierarchies):
    """
    Generalizes set valued series by flattening
    Parameters
    ----------
    series: Series
        Series to be recoded.
    recoding_rules: dict
        Dictionary containing recoding rules.
    hierarchies: dict
        Dictionary containing generalization hierarchies.
    Returns
    -------
    any
        Single value recoded to.
    """
    flattened, indexes, is_category = flatten_set_valued_series(series)
    if is_categorical_dtype(series) or is_category:
        flattened_series = pd.Series(flattened, index=indexes, dtype="category", name=series.name)
    else:
        flattened_series = pd.Series(flattened, index=indexes, name=series.name)
    result = recode(flattened_series, recoding_rules, hierarchies)
    return result.iloc[0]


def recode_strings(series):
    """
    Generalizes a series of strings by stepwise reduction of strings
    Parameters
    ----------
    series: Series
        Series to be recoded.
    Returns
    -------
    str
        Single value recoded to.
    """
    values = set(series.unique())
    while len(values) > 1:
        longest_element = next_string_to_reduce(values)
        values.remove(longest_element)
        generalized = reduce_string(longest_element)
        values.add(generalized)
    return list(values)[0]


def recode_range(series, hierarchy=None):
    """
    Generalizes a series of numbers to a range, using hierarchical brackets if provided
    Parameters
    ----------
    series: Series
        Series to be recoded.
    hierarchies: dict
        Generalization hierarchy.
    Returns
    -------
    range
        Single value recoded to.
    """
    if len(series.unique()) == 1:
        return series.unique().tolist()[0]
    if hierarchy:
        return recode_range_hierarchical(series, hierarchy)
    minimum = math.floor(min(series))
    maximum = math.ceil(max(series))
    return range(minimum, maximum + 1)


def recode_range_hierarchical(series, hierarchy):
    """
    Generalizes a series of numbers using hierarchical ranges
    Parameters
    ----------
    series: Series
        Series to be recoded.
    hierarchies: dict
        Generalization hierarchy.
    Returns
    -------
    AnyNode
        Single node covering all series items.
    """
    nodes_to_consider = list(hierarchy.leaves)
    nodes_to_consider.sort(key=lambda node: len(node.range))
    min_el = series.min()
    max_el = series.max()
    node = nodes_to_consider.pop(0)
    while not node.is_root:
        result = node.range
        if min_el in result and max_el in result:
            return node
        if node.parent not in nodes_to_consider:
            nodes_to_consider.append(node.parent)
            nodes_to_consider.sort(key=lambda node: len(node.range))
        node = nodes_to_consider.pop(0)
    return node


def recode_dates(series):
    """
    Generalizes a series of datetime objects by suppressing, day and month, and then generalizing to a range of years
    Parameters
    ----------
    series: Series
        Series to be recoded.
    Returns
    -------
    any
        Single value recoded to.
    """
    result = series.dt.normalize()
    if len(result.unique()) > 1:
        result = series.dt.to_period('M')
    if len(result.unique()) > 1:
        result = series.dt.to_period('Y')
    if len(result.unique()) > 1:
        years = series.apply(lambda x: x.year)
        years_range = recode_range(years)
        return years_range
    return result.tolist()[0]


def recode_ordinal(series):
    """
    Generalizes a series with ordered categorical values and returns either a single value (if all have the same), or a set of values
    Parameters
    ----------
    series: Series
        Series to be recoded.
    Returns
    -------
    any
        Single value recoded to or FrozenSet.
    """
    if not is_categorical_dtype(series) or not series.cat.ordered:
        raise Exception("Ordinal generalization cannot be applied to provided series")
    values = series.sort_values().unique().tolist()
    if len(values) == 1:
        return values[0]
    return frozenset(values)


def recode_nominal(series):
    """
    Generalizes a series with ordered categorical values and returns either a single value (if all have the same), or a set of values
    Parameters
    ----------
    series: Series
        Series to be recoded.
    Returns
    -------
    any
        Single value recoded to or FrozenSet.
    """
    if not is_categorical_dtype(series):
        raise Exception("Ordinal generalization cannot be applied to provided series")
    values = series.unique().tolist()
    if len(values) == 1:
        return values[0]
    return frozenset(values)


def recode_tokens(series):
    """
    Generalizes a series of SpaCy tokens by intersecting terms which appear
    Parameters
    ----------
    series: Series
        Series to be recoded.
    Returns
    -------
    list
        List containing remaining terms.
    """
    if series.isna().any():
        return None

    results = series.iloc[0]
    unique = series.iloc[0]
    for value in series[1:]:
        results, unique = intersect_token_lists(unique, value)
        if len(results) == 0:
            return None
    return results
