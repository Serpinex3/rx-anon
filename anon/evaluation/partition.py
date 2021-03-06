"""This module provides statistics on partitions"""
import logging
import numpy

logger = logging.getLogger(__name__)


def calculate_mean_partition_size(partitions):
    """
    Takes an array of partitions and calculates the mean partition size.
    Parameters
    ----------
    partitions: array
        Array containing the partitions.
    Returns
    -------
    float
        Mean partition size.
    """
    return numpy.mean(get_partition_lengths(partitions))


def calculate_std_partition_size(partitions):
    """
    Takes an array of partitions and calculates the standard deviation of the partition sizes.
    Parameters
    ----------
    partitions: array
        Array containing the partitions.
    Returns
    -------
    float
        Std for partition size.
    """
    return numpy.std(get_partition_lengths(partitions))


def get_partition_lengths(partitions):
    """
    Takes an array of partitions and returns and array with their lenghts.
    Parameters
    ----------
    partitions: array
        Array containing the partitions.
    Returns
    -------
    array
        Array containing partition lengths.
    """
    return [len(p) for p in partitions]


def get_partition_split_share(partition_split_statistics, textual_attribute_mapping):
    """
    Takes partition split statistics and a mapping for textual attributes and returns counts
    on splits of relational and textual attributes.
    Parameters
    ----------
    partition_split_statistics: dict
        Dictionary with attributes and numbers on splits applied on them.
    textual_attribute_mapping: dict
        Dictionary with textual attributes and their helper attributes.
    Returns
    -------
    tuple
        Tuple with relational splits and textual splits.
    """
    relational_splits = 0
    textual_splits = 0
    textual_attributes = []
    for key in textual_attribute_mapping:
        textual_attributes += textual_attribute_mapping[key]
    for attribute in partition_split_statistics:
        logger.debug("%d splits on attribute %s", partition_split_statistics[attribute], attribute)
        if attribute in textual_attributes:
            textual_splits += partition_split_statistics[attribute]
        else:
            relational_splits += partition_split_statistics[attribute]
    return relational_splits, textual_splits
