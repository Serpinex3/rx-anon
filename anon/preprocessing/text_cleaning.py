"""This modules contains functions to clean textual attributes"""
import re

from bs4 import BeautifulSoup


def remove_html_tags(text):
    """Removes HTML Tags from texts and replaces special spaces with regular spaces"""
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = text.replace(u'\xa0', ' ')
    return text


def remove_non_printable_characters(text):
    """Removes non-printable characters from the given text"""
    return re.sub(r'[^\x00-\x7F]+', '', text)


def remove_unnecessary_spaces(text):
    """Removes spaces at the beginning, end and multiple spaces in between in the given text"""
    return re.sub(' +', ' ', text).strip()
