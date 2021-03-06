"""This module contains code to prepare an RX-dataset for anonymization"""
import logging

from datetime import datetime

from nlp.similarity_module import compare_datetime, compare_using_equality
from tqdm import tqdm

from preprocessing.text_cleaning import remove_html_tags, remove_non_printable_characters, remove_unnecessary_spaces

logger = logging.getLogger(__name__)


class Preprocessor:
    """Stateful preprocessor which takes care of all prepraration to anonymize a dataset including detection of sensitive terms"""

    def __init__(self, ner, config, df):
        self.__ner = ner
        self.__config = config
        self.__df = df

        self.__relational_attributes = []
        self.__textual_attributes = []
        self.__non_redundant_entity_attributes = []
        self.__redundant_entity_attributes = []

        self.__prepare()

    def __prepare(self):
        self.__df.columns = map(str.lower, self.__df.columns)
        for attribute in self.__df.columns:
            anonymization_type = self.__config.attributes[attribute.lower()]["anonymization_type"]
            if (anonymization_type == "text"):
                self.__textual_attributes.append(attribute)
            else:
                self.__relational_attributes.append(attribute)

    def clean_textual_attributes(self):
        """
        Removes unprintable characters, HTML characters and unnecessary spaces from texts
        """
        for attribute in self.__textual_attributes:
            for index, text in self.__df[attribute].dropna().iteritems():
                text = remove_non_printable_characters(text)
                text = remove_html_tags(text)
                text = remove_unnecessary_spaces(text)
                self.__df.at[index, attribute] = text

    def analyze_textual_attributes(self):
        """
        Analyzes textual attributes on sensitive terms
        """
        for attribute in self.__textual_attributes:
            self.__analyze_textual_attribute(attribute)

    def __analyze_textual_attribute(self, textual_attribute):
        logger.info("Analyzing texts of attribute %s", textual_attribute)

        texts_to_analyze = {}
        for index, text in self.__df[textual_attribute].dropna().iteritems():
            record_id = self.__df[self.__config.get_key_attribute()][index]
            texts_to_analyze.setdefault(record_id, []).append((text, index))

        entities_per_record = self.__ner.recognize(textual_attribute, texts_to_analyze)
        logger.info("Converting results of analysis for further processing")
        for entity_type in self.__entities_to_consider():
            entities_found = []
            for index in tqdm(self.__df[textual_attribute].index, desc=entity_type):
                if index in entities_per_record:
                    entities = entities_per_record[index]
                    if entity_type in entities:
                        entities_found.append(list(entities[entity_type]))
                    else:
                        entities_found.append(None)
                else:
                    entities_found.append(None)
            non_redundant_attribute_name = self.__build_non_redundant_attribute_name(textual_attribute, entity_type)
            self.__df[non_redundant_attribute_name] = entities_found
            self.__non_redundant_entity_attributes.append(non_redundant_attribute_name)

    def get_non_redundant_entity_attributes(self):
        """
        Returns non-redundant entity columns.
        Returns
        -------
        list
            List containing non-redundant entity columns.
        """
        return self.__non_redundant_entity_attributes

    def get_redundant_entity_attributes(self):
        """
        Returns redundant entity columns.
        Returns
        -------
        list
            List containing redundant entity columns.
        """
        return self.__redundant_entity_attributes

    def __entities_to_consider(self):
        return set(self.__config.get_entities_to_consider()).intersection(set(self.__ner.get_recognized_entities()))

    def __build_non_redundant_attribute_name(self, attribute, entity_type):
        return "{}_{}".format(attribute, entity_type)

    def __build_redundant_attribute_name(self, attribute, entity_type):
        return "{}_{}_".format(attribute, entity_type)

    def find_redundant_information(self):
        """
        Resolves redundant information.
        """
        for attribute in self.__textual_attributes:
            self.__find_redundant_information(attribute)

    def compress(self):
        """
        Creates a person centric view of the dataset by grouping and aggregating based on the first direct identifier.
        """
        aggregation_functions = {}
        for attribute in self.__df.columns:
            aggregation_functions[attribute] = self.__aggregate
        grouped_df = self.__df.groupby(by=[self.__config.get_key_attribute()], as_index=False)
        self.__df = grouped_df.agg(aggregation_functions)
        self.__df = self.__df.astype(self.__config.get_data_types())

    def get_sensitive_terms(self):
        """
        Builds a dictionary containing terms and their appearances, categorized by entity type
        Returns
        -------
        list
            Dictionary containing terms and their appearances, categorized by entity type.
        """
        sensitive_terms_dict = {}
        for attribute in self.__non_redundant_entity_attributes:
            for record_id, sensitive_terms in self.__df[attribute].dropna().iteritems():
                for sensitive_term in sensitive_terms:
                    cleaned_sensitive_term = " ".join([t.lemma_.lower() for t in sensitive_term if not t.is_stop])
                    if len(cleaned_sensitive_term) > 0:
                        sensitive_terms_dict.setdefault(attribute, {}).setdefault(cleaned_sensitive_term, set()).add(record_id)

        # Sort sensitive terms dict alphabetically to have a deterministic order
        sensitive_terms_dict = {el[0]: el[1] for el in sorted(sensitive_terms_dict.items(), key=lambda x: x)}

        # Sort sensitive terms dict ascending by number terms per entity type
        sensitive_terms_dict = {el[0]: el[1] for el in sorted(sensitive_terms_dict.items(), key=lambda x: len(x[1]))}

        for attribute, sensitive_terms in sensitive_terms_dict.items():
            word = "terms"
            if len(sensitive_terms) == 1:
                word = "term"
            logger.info("Found %d distinct sensitive %s within attribute %s", len(sensitive_terms), word, attribute)
        return sensitive_terms_dict

    def get_df(self):
        """
        Returns the current status of the data frame.
        Returns
        -------
        DataFrame
            DataFrame with current state.
        """
        return self.__df

    def get_textual_attribute_mapping(self):
        """
        Returns a dictionary with the original textual attribute as key and the new temporary non redundant entity attributes as values.
        Returns
        -------
        dict
            Dictionary with textual attributes and non-redundant entity attributes as values.
        """
        textual_attributes = self.__config.get_textual_attributes()
        textual_attributes_mapping = {}
        for textual_attribute in textual_attributes:
            textual_attributes_mapping[textual_attribute] = [attribute for attribute in self.__non_redundant_entity_attributes if textual_attribute in attribute]
        return textual_attributes_mapping

    def __find_redundant_information(self, textual_attribute):
        logger.info("Looking for redundant information for attribute %s", textual_attribute)
        mapping = self.__config.get_attribute_entities_mapping()
        for entity_type in mapping:
            redundant_values = []
            redundant_found = False
            non_redundant_attribute_name = self.__build_non_redundant_attribute_name(textual_attribute, entity_type)
            if non_redundant_attribute_name not in self.__df.columns:
                logger.warning("Skipping entities of type %s for attribute %s since no entities exist.", entity_type, textual_attribute)
                continue

            for record_id in tqdm(self.__df.index, desc=entity_type):
                entities_of_record = self.__df.at[record_id, non_redundant_attribute_name]
                if entities_of_record:
                    redundant_information = self.__get_redundant_information(record_id, entities_of_record, mapping[entity_type])
                    if len(redundant_information) > 0:
                        redundant_found = True
                        redundant_values.append(list(redundant_information))
                        redundant_entities = set([e[0] for e in redundant_information])
                        non_redundant_values = list(set(self.__df.at[record_id, non_redundant_attribute_name]).difference(redundant_entities))  # Removing redundant terms from non redundant attribute
                        if len(non_redundant_values) == 0:
                            self.__df.at[record_id, non_redundant_attribute_name] = None  # Set to None if all sensitive terms are redundant
                        else:
                            self.__df.at[record_id, non_redundant_attribute_name] = non_redundant_values
                    else:
                        redundant_values.append(None)
                else:
                    redundant_values.append(None)

            if redundant_found:  # Create this series and extend dataframe
                redundant_attribute_name = self.__build_redundant_attribute_name(textual_attribute, entity_type)
                self.__redundant_entity_attributes.append(redundant_attribute_name)
                self.__df[redundant_attribute_name] = redundant_values

        self.__drop_empty_series()

        for redundant_entity_attribute in self.__redundant_entity_attributes:
            series = self.__df[redundant_entity_attribute].dropna()
            total = len(series.tolist())
            logger.info("Found redundant sensitive terms in %d records for attribute %s", total, redundant_entity_attribute)

    def __aggregate(self, series):
        """Used to aggregate data of a dataset, without losing any information."""
        if series.name in self.__non_redundant_entity_attributes or series.name in self.__redundant_entity_attributes:  # Textual entities
            merged_sensitive_terms = list()
            for sensitive_terms in series.dropna():
                merged_sensitive_terms = merged_sensitive_terms + sensitive_terms
            return merged_sensitive_terms if len(merged_sensitive_terms) > 0 else None  # Return merged result, or None
        else:
            if series.nunique() > 1:  # Since there are more values, pack them into a list / frozenset
                if series.name in self.__textual_attributes or series.name in self.__config.get_insensitive_attributes():
                    return list(series.array)
                else:
                    return frozenset(series.array)
            else:
                return series.unique()[0]  # Else return just this single value

    def __get_redundant_information(self, record_id, textual_values, relational_attributes):
        redundant_information = set()
        for attribute, relational_value in self.__df[relational_attributes].loc[record_id].iteritems():
            for textual_value in textual_values:
                if isinstance(relational_value, datetime):
                    result = compare_datetime(relational_value, textual_value)
                    right_matching_token = textual_value
                else:
                    nlp = self.__ner.get_nlp()
                    relational_token = nlp(str(relational_value))
                    result, _, right_matching_token = compare_using_equality(relational_token, textual_value)
                if result:
                    redundant_information.add((textual_value, right_matching_token, attribute))
        return redundant_information

    def __drop_empty_series(self):
        attributes_pre_drop = set(self.__df.columns)
        self.__df.dropna(axis=1, how='all', inplace=True)
        attributes_post_drop = set(self.__df.columns)
        attributes_dropped = attributes_pre_drop.difference(attributes_post_drop)
        if len(attributes_dropped) > 0:
            logger.warning("Dropped the attributes %s due to emptyness", ", ".join(attributes_dropped))
        self.__non_redundant_entity_attributes = set(self.__non_redundant_entity_attributes).intersection(attributes_post_drop)
