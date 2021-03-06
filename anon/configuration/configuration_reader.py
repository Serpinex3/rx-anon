"""Module containing corresponding code for reading a configuration file"""
import logging

from yaml import Loader, load

from .configuration import Configuration

logger = logging.getLogger(__name__)


class ConfigurationReader:
    """Class responsible for parsing a yaml config file and initializing a configuration"""

    def read(self, yaml_resource):
        """
        Reads a yaml config file and returns the corresponding configuration object
        Parameters
        ----------
        yaml_resource: (str, Path)
            The configuration file.
        Returns
        -------
        Configuration
            The corresponding configuration object
        """
        with open(yaml_resource, 'r') as file:
            doc = load(file, Loader=Loader)
            logger.info("Reading configuration from %s", yaml_resource)
            config = Configuration()

            # Make keys lower case to match columns in lower case
            config.attributes = {attr.lower(): v for attr, v in doc['attributes'].items()}

            if "entities" in doc:  # Overwrite default
                config.entities = doc['entities']
            if "parameters" in doc:  # Overwrite default
                config.parameters = doc['parameters']
            if "nlp" in doc:  # Overwrite default
                config.nlp = doc['nlp']

            logger.debug("Using the following configuration:\n%s", config)
            return config
