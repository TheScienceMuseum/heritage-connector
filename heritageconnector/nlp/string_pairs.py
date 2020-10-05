from fuzzywuzzy import fuzz
from typing import Union
from itertools import product


def fuzzy_match(
    string1: str, string2: str, scorer=fuzz.token_set_ratio, threshold=90
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


def fuzzy_match_lists(
    item1: Union[str, list],
    item2: Union[str, list],
    scorer=fuzz.token_set_ratio,
    threshold=90,
) -> bool:
    """
    Work out whether two strings or lists are similar to each other. Wrapper around `fuzzy_match`. 
    Takes the maximally similar item from the pair of lists.

    Args:
        item1 (Union[str, list])
        item2 (Union[str, list])
        scorer ([type], optional) Defaults to fuzz.token_set_ratio.
        threshold (int, optional) Defaults to 90.

    Returns:
        bool: True if the most similar pair are close matches
    """

    if isinstance(item1, str) and isinstance(item2, str):
        return fuzzy_match(item1, item2, scorer, threshold)

    else:
        if isinstance(item1, str):
            item1 = [item1]

        if isinstance(item2, str):
            item2 = [item2]

    for i1, i2 in product(item1, item2):
        if fuzzy_match(i1, i2, scorer, threshold):
            return True

    return False
