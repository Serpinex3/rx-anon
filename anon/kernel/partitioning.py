"""This module contains code for partitioning used to generate a k-anonymous view"""
import logging
import sys
import pandas as pd

from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_categorical_dtype

from kernel.util import flatten_set_valued_series, agg_mean, agg_categorical

logger = logging.getLogger(__name__)
sys.setrecursionlimit(3000)


def partition_mondrian(df, k, bias, relational_weight, quasi_identifiers):
    """
    Partitions a DataFrame in partitions with at least size k using Mondrian partitioning.
    Parameters
    ----------
    df: DataFrame
        DataFrame to be anonymized.
    k: int
        k, minimal group size.
    bias: dict
        Dictionary with attributes and their biases.
    relational_weight: float
        Tuning parameter for Mondrian.
    quasi_identifiers: list
        List with quasi-identifiers.
    Returns
    -------
    tuple
        Resulting partitions, and partition split statistics.
    """
    scale = __get_attribute_spans(df, df.index, quasi_identifiers)
    finished_partitions = []
    partitions = [df.index]
    partition_split_statistics = {attribute: 0 for attribute in quasi_identifiers}
    while partitions:
        partition = partitions.pop(0)
        if len(partition) >= 2 * k:
            logger.debug("Working on partition with length %d", len(partition))
            spans = __get_attribute_spans(df, partition, quasi_identifiers, scale)
            for column, _ in __mondrian_split_priority(spans, bias, relational_weight):
                lp, rp = __split_partition(df[column][partition])
                if not __is_k_anonymous(lp, k) or not __is_k_anonymous(rp, k):
                    continue
                if lp.equals(rp):
                    break
                else:
                    logger.debug("Splitting partition on attribute %s into two partitions with size %d and %d", column, len(lp), len(rp))
                    partition_split_statistics[column] += 1
                    partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        else:
            finished_partitions.append(partition)
        logger.debug("%d partitions remaining", len(partitions))
    return finished_partitions, partition_split_statistics


def partition_gdf(df, k, terms):
    """
    Partitions a DataFrame in partitions with at least size k using GDF partitioning.
    Parameters
    ----------
    df: DataFrame
        DataFrame to be anonymized.
    k: int
        k, minimal group size.
    terms: dict
        Dictionary with terms and records with their appearences.
    Returns
    -------
    array
        Resulting partitions.
    """
    return __partition_gdf_recursive(df, df.index, k, terms)


def __mondrian_split_priority(spans, bias, relational_weight):
    priority = {}
    textual_weight = 1 - relational_weight
    for attribute in spans:
        score = spans[attribute]
        score += bias.get(attribute, 0)
        if attribute in bias:
            score += (bias[attribute] + relational_weight)
        else:
            score += textual_weight
        priority[attribute] = score / 2
    priority = sorted(priority.items(), key=lambda x: -(x[1]))
    return priority


def __get_attribute_spans(df, partition, quasi_identifiers, scale=None):
    spans = {}
    for column in [col for col in quasi_identifiers if col in df.columns]:
        span = __get_attribute_span(df[column][partition])
        if scale is not None:
            span = span / scale[column]
        spans[column] = span
    return spans


def __get_attribute_span(series):
    if is_categorical_dtype(series):
        span = len(series.unique())
    elif is_datetime64_any_dtype(series):
        span = series.max() - series.min()
        span = span.days
    elif is_numeric_dtype(series):
        span = series.max() - series.min()
    else:
        flattened, indexes, is_category = flatten_set_valued_series(series)
        if is_category:
            new_series = pd.Series(flattened, dtype="category", index=indexes, name=series.name)
            new_series.index.name = "id"
            grouped = new_series.groupby(by="id").agg(agg_categorical).astype('category')
        else:
            new_series = pd.Series(flattened, index=indexes, name=series.name)
            new_series.index.name = "id"
            grouped = new_series.groupby(by="id").agg(agg_mean)
        span = __get_attribute_span(grouped)
    return span


def __split_partition(series):
    if is_categorical_dtype(series) or is_datetime64_any_dtype(series):
        values = series.sort_values().unique()
        lv = set(values[:len(values) // 2])
        rv = set(values[len(values) // 2:])
        return series.index[series.isin(lv)], series.index[series.isin(rv)]
    elif is_numeric_dtype(series):
        median = series.median()
        dfl = series.index[series < median]
        dfr = series.index[series >= median]
        return (dfl, dfr)
    else:
        flattened, indexes, is_category = flatten_set_valued_series(series)
        if is_category:
            new_series = pd.Series(flattened, index=indexes, dtype="category", name=series.name)
            new_series.index.name = "id"
            grouped = new_series.groupby(by="id").agg(agg_categorical).astype('category')
        else:
            new_series = pd.Series(flattened, index=indexes, name=series.name)
            new_series.index.name = "id"
            grouped = new_series.groupby(by="id").agg(agg_mean)
        return __split_partition(grouped)


def __partition_gdf_recursive(df, partition, k, terms):
    logger.debug("Working on partition with length %d", len(partition))
    if len(partition) <= k:
        return [partition]
    else:
        next_column, indexes, term = __next_entity_column(terms, partition, k)
        if next_column is None or indexes is None or term is None:
            return [partition]
        lp, rp = __split_textual_attribute(df, partition, next_column, indexes)
        if len(lp) == 0:
            terms[next_column].pop(term)
            return __partition_gdf_recursive(df, rp, k, terms)
        elif len(rp) == 0:
            terms[next_column].pop(term)
            return __partition_gdf_recursive(df, lp, k, terms)
        elif not __is_k_anonymous(lp, k) or not __is_k_anonymous(rp, k):
            return [partition]
        elif lp.equals(rp):
            return [lp]
        else:
            terms[next_column].pop(term)
            return __partition_gdf_recursive(df, lp, k, terms) + __partition_gdf_recursive(df, rp, k, terms)


def __next_entity_column(terms, partition, k):
    amount = 0
    r_category = None
    r_indexes = None
    term = None
    indexes = []
    for category in terms:
        temp = terms[category]
        filtered = {}
        temp_all = set()
        for key, value in temp.items():
            remaining = [ii for ii in value if ii in partition]
            temp_all.update(remaining)
            if len(remaining) >= k:
                filtered[key] = value
        normalizer = len(temp_all)
        for text in filtered:
            indexes = [ii for ii in terms[category][text] if ii in partition]
            new_amount = len(indexes) / normalizer
            if new_amount > amount:
                amount = new_amount
                r_category = category
                term = text
                r_indexes = indexes
    logger.debug("Splitting partition in category %s on term %s", r_category, term)
    return r_category, r_indexes, term


def __split_textual_attribute(df, partition, column, indexes):
    dfp = df[column][partition]
    remaining = set(partition).difference(set(indexes))
    return dfp.index[dfp.index.isin(indexes)], dfp.index[dfp.index.isin(remaining)]


def __is_k_anonymous(partition, k):
    return len(partition) >= k
