"""This module contains all code to create and work with configurations for the anon tool"""
import yaml
from anytree.importer import DictImporter
from anytree import LevelOrderIter

DEFAULT_BIAS = 0
DEFAULT_DATA_TYPE = 'nominal'
DEFAULT_ANONYMIZATION_TYPE = 'insensitive_attribute'
DEFAULT_DATE_FORMAT = '%d/%m/%Y'
DEFAULT_ORDER = None
DEFAULT_RECODING_STRATEGY = 'grouping'
DEFAULT_NLP_MODEL = "en_core_web_trf"
DEFAULT_NLP_CACHE = "data/cached_docs"
DEFAULT_K = 10
DEFAULT_STRATEGY = "mondrian"
DEFAULT_NATIVE_ENTITIES = []
DEFAULT_RELATIONAL_WEIGHT = 0.5

SUPPORTED_BIAS_LOWER_LIMIT = 0
SUPPORTED_BIAS_UPPER_LIMIT = 1
SUPPORTED_DATA_TYPES = ['nominal', 'ordinal', 'numerical', 'text', 'date']
SUPPORTED_ANONYMIZATION_TYPES = ['direct_identifier', 'quasi_identifier', 'insensitive_attribute', 'text']


class Configuration:
    """Class containing all code to configure the anonymization tool"""

    def __init__(self):
        self.parameters = {
            "k": DEFAULT_K,
            "strategy": DEFAULT_STRATEGY,
            "relational_weight": DEFAULT_RELATIONAL_WEIGHT  # Only used if strategy == "mondrian"
        }
        self.nlp = {
            "model": DEFAULT_NLP_MODEL,
            "cache": DEFAULT_NLP_CACHE
        }
        self.attributes = {}
        self.entities = {
            "native": DEFAULT_NATIVE_ENTITIES,
            "custom": {}
        }

    def __str__(self):
        config = {
            "parameters": self.parameters,
            "nlp": self.nlp,
            "attributes": self.attributes,
            "entities": self.entities
        }
        return yaml.dump(config)

    def get_direct_identifiers(self):
        """
        Returns a list of attribute names which are direct identifiers
        Returns
        -------
        list
            List of direct identifiers.
        """
        return [attribute for attribute in self.attributes if self.__get_anonymization_type(attribute) == 'direct_identifier']

    def get_quasi_identifiers(self):
        """
        Returns a list of attribute names which are quasi identifiers
        Returns
        -------
        list
            List of quasi identifiers.
        """
        return [attribute for attribute in self.attributes if self.__get_anonymization_type(attribute) == 'quasi_identifier']

    def get_insensitive_attributes(self):
        """
        Returns a list of attribute names which are insensitive
        Returns
        -------
        list
            List of insensitive attributes.
        """
        return [attribute for attribute in self.attributes if self.__get_anonymization_type(attribute) == 'insensitive_attribute']

    def get_textual_attributes(self):
        """
        Returns all textual attributes
        Returns
        -------
        list
            List of textual attributes.
        """
        return [attr for attr in self.attributes if self.__get_data_type(attr) == 'text']

    def get_date_attributes(self):
        """
        Returns all datetime attributes
        Returns
        -------
        list
            List of datetime attributes.
        """
        return [attr for attr in self.attributes if self.__get_data_type(attr) == 'date']

    def get_data_types(self):
        """
        Returns dict containing the attributes and their data types
        Returns
        -------
        dict
            Dictionary with attributes and their data types.
        """
        types_dict = {}
        for attribute in self.get_direct_identifiers() + self.get_quasi_identifiers() + self.get_textual_attributes():
            if self.__get_data_type(attribute) in ['nominal', 'ordinal']:
                types_dict[attribute] = 'category'
            elif self.__get_data_type(attribute) in ['text']:
                types_dict[attribute] = 'object'
        return types_dict

    def get_ordinal_orders(self):
        """
        Returns a dictionary containing ordinal attributes and their orders
        Returns
        -------
        dict
            Dictionary with ordinal attributes and their orders.
        """
        return {attr: self.__get_ordinal_order(attr) for attr in self.attributes if self.__get_data_type(attr) == 'ordinal' and self.__get_ordinal_order(attr) is not None}

    def get_biases(self):
        """
        Returns a dictionary containing quasi identifiers and their biases
        Returns
        -------
        dict
            Dictionary with quasi-identifying attributes and their biases.
        """
        return {attr: self.__get_bias(attr) for attr in self.get_quasi_identifiers()}

    def get_relational_weight(self):
        """
        Returns the weight of the relational attributes used for partitioning or the default
        Returns
        -------
        float
            Tuning parameter to control Mondrian partitioning.
        """
        return self.parameters.get("relational_weight", DEFAULT_RELATIONAL_WEIGHT)

    def get_date_formats(self):
        """
        Returns a dictionary containing datetime attributes and their date formats
        Returns
        -------
        dict
            Dictionary containing datetime attributes with their formats.
        """
        return {attr: self.__get_date_format(attr) for attr in self.get_date_attributes()}

    def get_default_date_format(self):
        """
        Returns the first default date format
        Returns
        -------
        str
            Default date format.
        """
        for attribute in self.get_date_attributes():
            return self.__get_date_format(attribute)
        return None

    def get_key_attribute(self):
        """
        Returns the first direct identifier as a key to group on
        Returns
        -------
        str
            Default key attribute.
        """
        for attribute in self.attributes:
            if self.__get_anonymization_type(attribute) == 'direct_identifier':
                return attribute
        return None

    def get_attribute_entities_mapping(self):
        """
        Returns a dictionary containing entities and their linked attributes to look for redundant information
        Returns
        -------
        dict
            Dictionary with the mapping of entity types and relational attributes.
        """
        mapping = {}
        for attribute_name in self.attributes:
            attribute = self.attributes[attribute_name]
            if "entities" in attribute:
                for entity in attribute["entities"]:
                    mapping.setdefault(entity, []).append(attribute_name)
        return mapping

    def get_recoding_strategy(self, attribute_name):
        """
        Returns the recoding strategy for the given attribute
        Parameters
        ----------
        attribute_name: str
            The attribute name.
        Returns
        -------
        str
            The recoding strategy.
        """
        attribute = self.attributes[attribute_name]
        if "recoding_strategy" in attribute:
            return attribute["recoding_strategy"]
        return DEFAULT_RECODING_STRATEGY

    def get_hierarchy(self, attribute_name):
        """
        Returns the hierarchy for a given attribute, None if there is no hierarchy
        Parameters
        ----------
        attribute_name: str
            The attribute name.
        Returns
        -------
        AnyNode
            The generalization hierarchy.
        """
        attribute = self.attributes[attribute_name]
        if "hierarchy" in attribute:
            importer = DictImporter()
            root = importer.import_(attribute['hierarchy'])
            for node in LevelOrderIter(root):
                node_range = name_to_range(node.name)
                node.range = node_range
            return root
        return None

    def get_entities_to_consider(self):
        """
        Returns all entities to consider during text anonymization
        Returns
        -------
        list
            List of entity types which are used for text anonymization.
        """
        entities = self.entities["native"]
        if "custom" in self.entities.keys():
            entities += self.entities["custom"].keys()
        return entities

    def __get_data_type(self, attribute):
        if attribute in self.attributes:
            return self.attributes[attribute].setdefault('type', DEFAULT_DATA_TYPE)
        return DEFAULT_DATA_TYPE

    def __get_bias(self, attribute):
        if attribute in self.attributes:
            bias = self.attributes[attribute].setdefault('bias', DEFAULT_BIAS)
            if not is_supported_bias(bias):
                raise Exception("Invalid bias {} for attribute {}. Bias must be between 0 and 1.".format(bias, attribute))
            return bias
        return DEFAULT_BIAS

    def __get_date_format(self, attribute):
        if attribute in self.attributes:
            return self.attributes[attribute].setdefault('format', DEFAULT_DATE_FORMAT)
        return DEFAULT_DATE_FORMAT

    def __get_ordinal_order(self, attribute):
        if attribute in self.attributes:
            return self.attributes[attribute].setdefault('order', DEFAULT_ORDER)
        return DEFAULT_ORDER

    def __get_anonymization_type(self, attribute):
        if attribute in self.attributes:
            return self.attributes[attribute].setdefault('anonymization_type', DEFAULT_ANONYMIZATION_TYPE)
        return DEFAULT_ANONYMIZATION_TYPE


def name_to_range(name):
    """Converts a name of numerical hierarchy to a range"""
    start_stop = name.split("-")
    return range(int(start_stop[0]), int(start_stop[1]) + 1)


def is_supported_bias(bias):
    """Returns true if bias is in supported range"""
    return SUPPORTED_BIAS_LOWER_LIMIT <= bias <= SUPPORTED_BIAS_UPPER_LIMIT


def is_supported_data_type(arg):
    """Returns true if data type is supported"""
    return arg in SUPPORTED_DATA_TYPES


def is_supported_anonymization_type(arg):
    """Returns true if anonymization type is supported"""
    return arg in SUPPORTED_ANONYMIZATION_TYPES
