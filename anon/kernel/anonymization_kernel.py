"""This module contains code to anonymize a dataset wrapping around k-anonymity"""
import logging

from kernel.k_anonymity import KAnonymity

logger = logging.getLogger(__name__)


class AnonymizationKernel:
    """
    Anonymization kernel which takes a configuration, the term frequency distribution,
    the named entity recognition module and the preprocessor.
    """

    def __init__(self, terms, config, ner, pp):
        self.__terms = terms
        self.__config = config
        self.__ner = ner
        self.__preprocessor = pp

    def anonymize_quasi_identifiers(self, df, k=None, strategy=None, biases=None, relational_weight=None):
        """
        Anonymizes quasi-identifying attributes as well as sensitive information in texts by applying k-anonymity
        Parameters
        ----------
        df: DataFrame
            DataFrame to be anonymized.
        k: int
            k, minimal group size.
        strategy: str
            Partitioning strategy.
        biases: dict
            Dictionary with attributes and their biases.
        relational_weight: float
            Tuning parameter for Mondrian.
        Returns
        -------
        tuple
            Anonymized DataFrame, resulting partitions, and partition split statistics.
        """
        if not k:
            k = self.__config.parameters["k"]
        if not strategy:
            strategy = self.__config.parameters["strategy"]
        if not biases:
            biases = self.__config.get_biases()
        if relational_weight is None and relational_weight != 0:
            relational_weight = self.__config.get_relational_weight()
        return self.__apply_k_anonymity(df.copy(), k, strategy, biases, relational_weight)

    def remove_direct_identifier(self, df):
        """
        Removes direct identifiers given a dataframe
        Parameters
        ----------
        df: DataFrame
            DataFrame to be anonymized.
        Returns
        -------
        DataFrame
            Anonymized DataFrame.
        """
        direct_identifiers = self.__config.get_direct_identifiers()
        df = df.drop(columns=direct_identifiers)
        logger.info("Dropped direct identifying attributes %s", ", ".join(direct_identifiers))
        return df

    def __apply_k_anonymity(self, df, k, strategy, bias, relational_weight):
        quasi_identifiers = self.__config.get_quasi_identifiers()
        k_anonymity = KAnonymity(df, quasi_identifiers, k, strategy, bias, relational_weight, self.__terms, self.__config)
        anonymized_df, partitions, partition_split_statistics = k_anonymity.anonymize()
        for col in anonymized_df.columns:
            df[col] = anonymized_df[col]
        return df, partitions, partition_split_statistics

    def recode_textual_attributes(self, df):
        """
        Takes a dataframe and recodes sensitive terms appearing in its texts by their anonymized representatives
        Parameters
        ----------
        df: DataFrame
            DataFrame to be anonymized.
        Returns
        -------
        DataFrame
            Anonymized DataFrame.
        """
        for attr in self.__config.get_textual_attributes():
            df = self.__recode_sensitive_terms(df, attr)
        return df

    def __recode_sensitive_terms(self, df, text_attribute):
        for index in df[text_attribute].dropna().index:
            entities_to_remain = set()
            replacements = dict()
            for entity_column in self.__preprocessor.get_non_redundant_entity_attributes():
                entity_to_remain = df[entity_column][index]
                if entity_to_remain:
                    entities_to_remain = entities_to_remain.union(entity_to_remain)
            for redundant_entity_column in self.__preprocessor.get_redundant_entity_attributes():
                entity_to_remain = df[redundant_entity_column][index]
                if entity_to_remain:
                    for (entity, token, col) in entity_to_remain:
                        replacements.setdefault(entity.label_, []).append((entity, token, df.at[index, col]))
            df.at[index, text_attribute] = self.__ner.replace(text_attribute, df[self.__config.get_key_attribute()][index], replacements, entities_to_remain)
        return df
