"""This module includes tests for reading configurations"""
from unittest import TestCase

from configuration.configuration_reader import ConfigurationReader


class TestConfigurationReader(TestCase):
    """This class contains tests for the configuration reader"""

    def test_read_of_configuration(self):
        configuration_reader = ConfigurationReader()
        config = configuration_reader.read('./tests/resources/sample_config.yaml')
        self.assertIsNotNone(config)
