"""This module contains tests for evaluating similarity"""
import datetime
from unittest import TestCase

import spacy
from nlp.similarity_module import compare_datetime, compare_using_equality

model = 'en_core_web_trf'

nlp = spacy.load(model)


class TestSimilarityModule(TestCase):
    """This class contains tests for the similarity module"""

    def test_equality(self):
        doc_1 = nlp("I live in Munich")
        doc_2 = nlp("I love Munich!")
        self.assertTrue(compare_using_equality(doc_1.ents[0], doc_2.ents[0]))

    def test_datetime(self):
        doc = nlp("I was born in 2005!")
        date = datetime.date(2005, 3, 17)
        self.assertTrue(compare_datetime(date, doc.ents[0]))
