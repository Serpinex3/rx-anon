"""This module contains tests for recoding"""

from unittest import TestCase
import datetime
import pandas as pd

from kernel.recoding import recode, recode_dates, recode_ordinal, recode_nominal, recode_range
from kernel.util import reduce_string


class TestStringGeneralization(TestCase):
    """Class containing tests for string generalization"""

    def test_generalization(self):
        postcode = 'NE9 5YE'
        generalized = reduce_string(postcode)
        self.assertNotEqual(postcode, generalized)

    def test_single_step_generalization(self):
        postcode_1 = 'HP2 7PW'
        postcode_2 = 'HP2 7PF'
        generalized_1 = reduce_string(postcode_1)
        generalized_2 = reduce_string(postcode_2)

        self.assertNotEqual(postcode_1, postcode_2)
        self.assertEqual(generalized_1, generalized_2)

    def test_multistep_generalization(self):
        postcode_1 = 'HP2 7PW'
        postcode_2 = 'HP2 4DY'

        number_of_generalization_steps = 0

        while(postcode_1 != postcode_2):
            if (len(postcode_1) > len(postcode_2)):
                postcode_1 = reduce_string(postcode_1)
            else:
                postcode_2 = reduce_string(postcode_2)
            number_of_generalization_steps = number_of_generalization_steps + 1
        self.assertEqual(postcode_1, postcode_2)
        self.assertEqual(number_of_generalization_steps, 6)

    def test_total_generalization(self):
        postcode_1 = 'HP2 7PW'
        postcode_2 = 'CF470JD'

        number_of_generalization_steps = 0

        while(postcode_1 != postcode_2):
            if (len(postcode_1) > len(postcode_2)):
                postcode_1 = reduce_string(postcode_1)
            else:
                postcode_2 = reduce_string(postcode_2)
            number_of_generalization_steps = number_of_generalization_steps + 1

        self.assertEqual(postcode_1, postcode_2)
        self.assertEqual(number_of_generalization_steps, 14)
        self.assertEqual(postcode_1, '*')


class TestRangeGeneralization(TestCase):
    """Class containing tests for range generalization"""

    def test_range_of_ints_generalization(self):
        numbers = [2, 5, 27, 12, 3]
        generalized = recode_range(pd.Series(numbers))

        self.assertIsInstance(generalized, range)
        self.assertEqual(generalized, range(2, 28))

    def test_range_of_floats_generalization(self):
        numbers = [8.7, 4.12, 27.3, 18]
        generalized = recode_range(pd.Series(numbers))
        self.assertIsInstance(generalized, range)
        self.assertEqual(generalized, range(4, 29))


class TestDateGeneralization(TestCase):
    """Class containing tests for date generalization"""

    def test_time_generalization(self):
        date_1 = datetime.datetime(2020, 9, 28, 12, 32, 00)
        date_2 = datetime.datetime(2020, 9, 28, 15, 27, 48)

        series = pd.Series([date_1, date_2])
        generalized = recode_dates(series)

        self.assertEqual(generalized, datetime.datetime(2020, 9, 28))

    def test_day_generalization(self):
        date_1 = datetime.datetime(2020, 9, 27, 12, 32, 00)
        date_2 = datetime.datetime(2020, 9, 28, 15, 27, 48)

        series = pd.Series([date_1, date_2])
        generalized = recode_dates(series)
        self.assertEqual(generalized.to_timestamp(), datetime.datetime(2020, 9, 1))

    def test_month_generalization(self):
        date_1 = datetime.datetime(2020, 10, 27, 12, 32, 00)
        date_2 = datetime.datetime(2020, 9, 28, 15, 27, 48)

        series = pd.Series([date_1, date_2])
        generalized = recode_dates(series)
        self.assertEqual(generalized.to_timestamp(), datetime.datetime(2020, 1, 1))

    def test_year_generalization(self):
        date_1 = datetime.datetime(2021, 10, 27, 12, 32, 00)
        date_2 = datetime.datetime(2020, 9, 28, 15, 27, 48)

        series = pd.Series([date_1, date_2])
        generalized = recode_dates(series)
        self.assertEqual(generalized, range(2020, 2022))


class TestOrdinalGeneralization(TestCase):
    """Class containing tests for ordinal generalization"""

    def test_ordinal_generalization_raises_exception(self):
        categories = ['A', 'B', 'C']
        values = ['A', 'A', 'A']
        series = pd.Series(pd.Categorical(values, categories, ordered=False))
        self.assertRaises(Exception, recode_ordinal, series)

    def test_ordinal_generalization_with_single_category(self):
        categories = ['A', 'B', 'C']
        values = ['A', 'A', 'A']
        series = pd.Series(pd.Categorical(values, categories, ordered=True))
        generalized = recode_ordinal(series)
        self.assertEqual(generalized, 'A')

    def test_ordinal_generalization_with_multiple_categories(self):
        categories = set(['A', 'B', 'C'])
        values = ['B', 'A', 'B', 'C', 'A']
        series = pd.Series(pd.Categorical(values, categories, ordered=True))
        generalized = recode_ordinal(series)
        self.assertSetEqual(generalized, categories)


class TestNominalGeneralization(TestCase):
    """Class containing tests for nominal generalization"""

    def test_nominal_generalization_raises_exception(self):
        values = ['A', 'A', 'A']
        series = pd.Series(values)
        self.assertRaises(Exception, recode_nominal, series)

    def test_nominal_generalization_with_single_value(self):
        categories = ['A', 'B', 'C']
        values = ['A', 'A', 'A']
        series = pd.Series(pd.Categorical(values, categories, ordered=False))
        generalized = recode_nominal(series)
        self.assertEqual(generalized, 'A')

    def test_nominal_generalization_with_multiple_values(self):
        categories = set(['A', 'B', 'C'])
        values = ['B', 'A', 'B', 'C', 'A']
        series = pd.Series(pd.Categorical(values, categories, ordered=False))
        generalized = recode_nominal(series)
        self.assertSetEqual(categories, generalized)


class TestGeneralization(TestCase):
    """Class containing tests for main entry point in generalization"""

    def test_date_generalization(self):
        date_1 = datetime.datetime(2021, 10, 27, 12, 32, 00)
        date_2 = datetime.datetime(2020, 9, 28, 15, 27, 48)

        series = pd.Series([date_1, date_2])
        series = recode(series)
        self.assertEqual(len(series.unique()), 1)
