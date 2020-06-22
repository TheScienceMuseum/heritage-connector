from fuzzywuzzy import fuzz


def fuzzy_match(
    string1: str, string2: str, scorer=fuzz.token_sort_ratio, threshold=90
) -> bool:
    """
    Work out whether two strings are similar to each other.

    Args:
        string1 (str), string2(str): the two strings to compare
        scorer (func): a function which takes two strings and returns a similarity measure between 0 and 100
        threshold (int): a threshold above which two strings are considered similar. Between 0 and 100.

    Returns:
        bool: True if strings are close matches
    """

    assert 0 <= threshold <= 100

    result = scorer(string1, string2)

    return result >= threshold
