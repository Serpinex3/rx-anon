"""This modules contains code to apply k-anonymity on datasets"""
import logging
import pandas as pd
from kernel.recoding import recode
from tqdm import tqdm

from kernel.partitioning import partition_mondrian, partition_gdf

logger = logging.getLogger(__name__)


class KAnonymity:

    def __init__(self, df, quasi_identifiers, k, strategy, bias, relational_weight, terms, config):
        self.__k = k
        self.__quasi_identifiers = quasi_identifiers
        self.__df = df
        self.__bias = bias
        self.__terms = terms
        self.__strategy = strategy
        self.__relational_weight = relational_weight
        self.__config = config

    def anonymize(self):
        """
        Anonymizes data frame by first partitioning the data and later recoding the partitions according to predefined parameters
        Returns
        -------
        tuple
            Anonymized DataFrame, resulting partitions, and partition split statistics.
        """
        partition_split_statistics = None
        if self.__strategy == "mondrian":
            if self.__relational_weight == 0:
                ordered_quasi_identifiers = list(self.__terms.keys())  # Relational attributes are ignored during partitioning
            elif self.__relational_weight == 1:
                ordered_quasi_identifiers = self.__quasi_identifiers  # Textual attributes are ignored during partitioning
            elif self.__relational_weight >= 0.5:
                ordered_quasi_identifiers = self.__quasi_identifiers + list(self.__terms.keys())  # Use both, but put relational attributes up front
            else:
                ordered_quasi_identifiers = list(self.__terms.keys()) + self.__quasi_identifiers  # Use both, but put textual attributes up front
            logger.info("Partition dataset using %s on attributes %s with k=%d", self.__strategy, ", ".join(ordered_quasi_identifiers), self.__k)

            # partition using mondrian
            finished_partitions, partition_split_statistics = partition_mondrian(self.__df, self.__k, self.__bias, self.__relational_weight, ordered_quasi_identifiers)
        elif self.__strategy == "gdf":
            # partition using gdf
            finished_partitions = partition_gdf(self.__df, self.__k, self.__terms)
        else:
            raise Exception("Partitioning strategy {} no supported".format(self.__strategy))

        # Recode dataset to get a k-anonymous version
        anonymized_df = self.__recode(finished_partitions)

        # Return anonymized dataset and partitions
        return anonymized_df, finished_partitions, partition_split_statistics

    def __recode(self, partitions):
        # Set up hierarchies and recoding rules
        hierarchies, recoding_rules = self.__get_recoding_parameters()

        # Determine attributes to recode
        attributes_to_recode = [a for a in self.__quasi_identifiers + list(self.__terms.keys()) if a in self.__df.columns]

        recoded_df = pd.DataFrame()
        for partition in tqdm(partitions, desc="Recoding"):
            sub_frame = self.__df.loc[partition, attributes_to_recode]
            recoded_sub_frame = sub_frame.transform(recode, recoding_rules=recoding_rules, hierarchies=hierarchies)
            recoded_df = pd.concat([recoded_df, recoded_sub_frame])
        return recoded_df

    def __get_recoding_parameters(self):
        hierarchies = {}
        recoding_rules = {}
        for column in self.__quasi_identifiers:
            recoding_strategy = self.__config.get_recoding_strategy(column)
            recoding_rules[column] = recoding_strategy
            hierarchy = self.__config.get_hierarchy(column)
            if hierarchy:
                hierarchies[column] = hierarchy
        return hierarchies, recoding_rules
