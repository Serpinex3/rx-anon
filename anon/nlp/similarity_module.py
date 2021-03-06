"""This module contains methods to compare information based on similarity"""


def compare_using_equality(span1, span2):
    """
    Compare two spans and return true, if lower words in those spans are equal (stop words excluded)
    Parameters
    ----------
    span1: Span
        First span to compare.
    span2: Span
        Second span to compare.
    Returns
    -------
    tuple
        Tuple with a flag indicating equality, and tokens of both spans matching
    """
    for first_token in [t for t in span1 if not t.is_stop]:
        for second_token in [t for t in span2 if not t.is_stop]:
            if first_token.lemma_.lower() == second_token.lemma_.lower():
                return True, first_token, second_token
    return False, None, None


def compare_complete_match(span1, span2):
    """
    Compare two spans and return true, if lower words in those spans are equal (stop words excluded)
    Parameters
    ----------
    span1: Span
        First span to compare.
    span2: Span
        Second span to compare.
    Returns
    -------
    bool
        True if spans match.
    """
    if " ".join([t.lemma_.lower() for t in span1 if not t.is_stop]) == " ".join([a.lemma_.lower() for a in span2 if not a.is_stop]):
        return True
    return False


def compare_datetime(date, span):
    """
    Compare information within datetime object with a span
    Parameters
    ----------
    date: datetime
        Datetime to compare.
    span: Span
        Span to compare.
    Returns
    -------
    bool
        True if match.
    """
    return span.text in str(date)
