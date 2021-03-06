"""This module contains code to calculate the NCP for heterogeneous datasets"""
import logging
import math
import pandas as pd
from anytree import AnyNode
from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype
from kernel.util import flatten_set_valued_series, is_node, is_token_list, must_be_flattened
from kernel.recoding import recode_range_hierarchical

logger = logging.getLogger(__name__)


def calculate_normalized_certainty_penalty(original, anonymized, relational_attributes, textual_attributes_mapping):
    """
    Takes the original dataset, the anonymized dataset, a list or relational quasi-identifying attributes,
    and textual attributes and calculates the Normalized Certainty Penalty (NCP).
    Parameters
    ----------
    original: DataFrame
        The orgininal dataframe.
    anonymized: DataFrame
        The anonymized dataframe.
    relational_attributes: list
        List containing relational attributes.
    textual_attributes_mapping: dict
        Mapping of textual attributes and their helper attributes.
    Returns
    -------
    Tuple
        Tuple with total information loss, relational information loss, and detailed textual information loss.
    """
    # Calculate relation information loss
    relational_information_loss = 0
    for attribute in [attr for attr in original if attr in relational_attributes]:
        ncp = __calculate_ncp_attribute(original[attribute], anonymized[attribute])
        relational_information_loss = relational_information_loss + ncp
        logger.debug("Information loss for attribute %s is %4.4f", attribute, ncp)

    relational_information_loss = relational_information_loss / len(relational_attributes)

    # Calculate textual information loss
    if len(textual_attributes_mapping) > 0:
        textual_information_loss = {}
        total_loss = 0

        # For each original textual attribute
        for mapping in textual_attributes_mapping:
            textual_attributes = textual_attributes_mapping[mapping]
            textual_information_loss[mapping] = {}
            original_textual_tokens = []
            anonymized_textual_tokens = []

            # Individual textual information loss per attribute and entity
            for attribute in textual_attributes:
                attribute_loss = __calculate_ncp_attribute(original[attribute], anonymized[attribute])
                textual_information_loss[mapping][attribute] = attribute_loss
                logger.debug("Information loss for entity type %s is %4.4f", attribute, attribute_loss)

            # Total textual information loss per attribute
            for index in original.index:
                original_container = []
                anonymized_container = []
                for attribute in textual_attributes:
                    original_for_col = original.at[index, attribute]
                    anonymized_for_col = anonymized.at[index, attribute]
                    if original_for_col:
                        original_container += original_for_col
                    if anonymized_for_col:
                        anonymized_container += anonymized_for_col
                if len(original_container) > 0:
                    original_textual_tokens.append(original_container)
                else:
                    original_textual_tokens.append(None)
                if len(anonymized_container) > 0:
                    anonymized_textual_tokens.append(anonymized_container)
                else:
                    anonymized_textual_tokens.append(None)
            attribute_total_loss = __calculate_ncp_attribute(pd.Series(original_textual_tokens, index=original.index), pd.Series(anonymized_textual_tokens, index=anonymized.index))

            # Set total information loss for a single attribute
            textual_information_loss[mapping]["total"] = attribute_total_loss
            logger.debug("Information loss for attribute %s is %4.4f", mapping, attribute_total_loss)
            total_loss += attribute_total_loss

        # Set total information loss for all textual attributes
        textual_information_loss["total"] = total_loss / len(textual_attributes_mapping)
        return (relational_information_loss + textual_information_loss["total"]) / 2, relational_information_loss, textual_information_loss
    return relational_information_loss, relational_information_loss, None


def __calculate_ncp_attribute(original_series, anonymized_series):
    if must_be_flattened(original_series):
        original_flattened, original_indexes, is_category = flatten_set_valued_series(original_series)
        if is_categorical_dtype(original_series) or is_category:
            original_flattened_series = pd.Series(original_flattened, index=original_indexes, dtype="category", name=original_series.name)
        else:
            original_flattened_series = pd.Series(original_flattened, index=original_indexes, name=original_series.name)
        ncp = __calculate_ncp_attribute(original_flattened_series, anonymized_series)
    elif is_node(anonymized_series):  # Has been anonymized using a hierarchy
        ncp = __ncp_numerical_hierarchy(original_series, anonymized_series)
    elif is_datetime64_any_dtype(original_series):
        ncp = __ncp_date(original_series, anonymized_series)
    elif is_categorical_dtype(original_series):
        ncp = __ncp_categorical(original_series, anonymized_series)
    elif is_numeric_dtype(original_series):
        ncp = __ncp_numerical(original_series, anonymized_series)
    elif is_token_list(original_series):
        ncp = __ncp_tokens(original_series, anonymized_series)
    else:
        ncp = __ncp_set_valued(original_series, anonymized_series)
    return ncp


def __ncp_numerical(original_series, anonymized_series):
    orig_min = math.floor(original_series.min())
    orig_max = math.ceil(original_series.max())
    orig_range = orig_max - orig_min + 1
    acc_information_loss = 0
    for value in anonymized_series:
        if not isinstance(value, range):
            pass
        else:
            anonymized_range = len(value)
            acc_information_loss = acc_information_loss + (anonymized_range / orig_range)
    normalized_information_loss = acc_information_loss / len(anonymized_series)
    return normalized_information_loss


def __ncp_numerical_hierarchy(original_series, anonymized_series):
    for value in anonymized_series:
        if isinstance(value, AnyNode):
            hierarchy = value.root
            break
    worst_generalization = recode_range_hierarchical(original_series, hierarchy)
    worst_generalization_range = len(worst_generalization.range)
    acc_information_loss = 0
    for value in anonymized_series:
        if not isinstance(value, AnyNode):
            pass
        else:
            anonymized_range = len(value.range)
            acc_information_loss = acc_information_loss + (anonymized_range / worst_generalization_range)
    normalized_information_loss = acc_information_loss / len(anonymized_series)
    return normalized_information_loss


def __ncp_categorical(original_series, anonymized_series):
    orig_n_categories = len(original_series.unique())
    acc_information_loss = 0
    for value in anonymized_series:
        if isinstance(value, frozenset):  # Otherwise it is only one value -> ncp of 0
            anon_n_categories = len(value)
            acc_information_loss = acc_information_loss + (anon_n_categories / orig_n_categories)
    normalized_information_loss = acc_information_loss / len(anonymized_series)
    return normalized_information_loss


def __ncp_set_valued(original_series, anonymized_series):
    original_flattened, original_indexes, _ = flatten_set_valued_series(original_series)
    anonymized_flattened, anonymized_indexes, _ = flatten_set_valued_series(anonymized_series)
    if is_categorical_dtype(original_series):
        original_flattened_series = pd.Series(original_flattened, index=original_indexes, dtype="category", name=original_series.name)
    else:
        original_flattened_series = pd.Series(original_flattened, index=original_indexes, name=original_series.name)

    if is_categorical_dtype(anonymized_series):
        anonymized_flattened_series = pd.Series(anonymized_flattened, index=anonymized_indexes, dtype="category", name=anonymized_series.name)
    else:
        anonymized_flattened_series = pd.Series(anonymized_flattened, index=anonymized_indexes, name=anonymized_series.name)

    return __calculate_ncp_attribute(original_flattened_series, anonymized_flattened_series)


def __ncp_date(original_series, anonymized_series):
    orig_unique_dates = len(original_series.unique())
    acc_information_loss = 0
    for value in anonymized_series:
        if isinstance(value, range):
            anon_unique_dates = 0
            for year in value:
                anon_unique_dates = anon_unique_dates + len(original_series[original_series.dt.year == year].unique())
            acc_information_loss = acc_information_loss + (anon_unique_dates / orig_unique_dates)
        elif isinstance(value, pd.Period):
            start_date = value.start_time.normalize()
            end_date = value.end_time.normalize()
            original_series = original_series.dt.normalize()
            anon_unique_dates = len(original_series.loc[(
                original_series >= start_date) & (original_series <= end_date)].unique())
            acc_information_loss = acc_information_loss + (anon_unique_dates / orig_unique_dates)
    normalized_information_loss = acc_information_loss / len(anonymized_series)
    return normalized_information_loss


def __ncp_tokens(original_series, anonymized_series):
    acc_information_loss = 0
    original_series = original_series.dropna()
    anonymized_series = anonymized_series.dropna()
    for index in original_series.index:
        original = original_series[index]
        if index not in anonymized_series.index:
            acc_information_loss = acc_information_loss + 1  # Since no entities remain
        else:
            anonymized = anonymized_series[index]
            anonymized = set(anonymized).intersection(set(original))
            acc_information_loss = acc_information_loss + (1 - (len(anonymized) / len(original)))
    normalized_information_loss = acc_information_loss / len(original_series)
    return normalized_information_loss
