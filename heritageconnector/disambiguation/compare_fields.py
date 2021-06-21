from fuzzywuzzy import fuzz
from typing import Union
from itertools import product
import numpy as np
import rdflib
import re

from heritageconnector.namespace import WD
from heritageconnector.utils.wikidata import url_to_qid, year_from_wiki_date
from heritageconnector import logging

logger = logging.get_logger(__name__)

# Similarity measures to compare fields of different types
# Each function should use the following template:
# ```
# def similarity_<type>(val1, val2, **kwargs) -> float:
#   ...
#   # 0 <= sim <= 1
#   return sim
# ```


def compare(
    internal_val: Union[rdflib.Literal, rdflib.URIRef],
    wikidata_entity: str,
    wikidata_label: str,
):
    """
    High-level comparison function for a pair of values. Determines which similarity measure to use
    depending on the type of `internal_val` and trying to coerce literal values to different python
    types.

    If a list passed to `internal_val` the method uses the first truthy element of the list to determine
    which similarity measure to use, then passes the whole list to the similarity function.

    Args:
        internal_val (Union[rdflib.Literal, rdflib.URIRef]): value from Heritage Connector graph.
        wikidata_entity (str): Wikidata entity (QID or value)
        wikidata_label (str): label of Wikidata entity
    """

    if isinstance(internal_val, list):
        if not all(
            [isinstance(val, (rdflib.Literal, rdflib.URIRef)) for val in internal_val]
        ):
            raise ValueError(
                f"Input value internal_val must be either rdflib.Literal or rdflib.URIRef ({type(internal_val)} passed)"
            )
        list_input = True
        internal_val_test = [i for i in internal_val if bool(i)][0]
    else:
        if not isinstance(internal_val, (rdflib.Literal, rdflib.URIRef)):
            raise ValueError(
                f"Input value internal_val must be either rdflib.Literal or rdflib.URIRef ({type(internal_val)} passed)"
            )
        list_input = False
        internal_val_test = internal_val

    # convert wikidata_entity from date to integer if it's date-like
    # if not, wikidata_entity keeps the same value
    wikidata_entity = year_from_wiki_date(wikidata_entity)

    if isinstance(internal_val_test, rdflib.URIRef) and str(
        internal_val_test
    ).startswith(str(WD)):
        # Wikidata entity: categorical comparison between entities
        val_as_qid = (
            url_to_qid(str(internal_val))
            if not list_input
            else [
                url_to_qid(str(item))
                for item in internal_val
                if type(item) == type(internal_val_test)
            ]
        )

        return similarity_categorical(
            val_as_qid, wikidata_entity, raise_on_diff_types=False
        )

    elif isinstance(internal_val_test, rdflib.Literal):
        try:
            # value is numeric
            float(internal_val_test)

            val_as_numeric = (
                float(internal_val)
                if not list_input
                else [float(item) for item in internal_val]
            )
            return similarity_numeric(val_as_numeric, wikidata_entity)

        except ValueError:
            # assume value is string: compare with label
            val_as_string = (
                str(internal_val)
                if not list_input
                else [str(item) for item in internal_val]
            )

            return similarity_string(val_as_string, wikidata_label)


def similarity_string(
    val1: Union[str, list], val2: Union[str, list], scorer=fuzz.token_set_ratio
) -> float:
    """
    Calculate string similarity. If val1 and val2 are lists or tuples, the similarity of the most similar pair
        of values between the lists is returned.

    Args:
        val1 (Union[str, list])
        val2 (Union[str, list])
        scorer (optional): Takes two strings and outputs an integer between 1 and 100.
            Defaults to fuzz.token_set_ratio.

    Returns:
        float: 0 <= f <= 1
    """

    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    if isinstance(val1, str) and isinstance(val2, str):
        return scorer(val1, val2) / 100

    elif (isinstance(val1, str) or isinstance(val2, str)) and (
        isinstance(val1, list) or isinstance(val2, list)
    ):
        # one is a string and one is a list

        str_val = [i for i in [val1, val2] if isinstance(i, (str))][0]
        list_val = [i for i in [val1, val2] if isinstance(i, (list, tuple))][0]

        return max([scorer(str_val, i) for i in list_val]) / 100

    elif isinstance(val1, list) and isinstance(val2, list):

        return max(scorer(i[0], i[1]) for i in product(val1, val2)) / 100


def similarity_numeric(
    val1: Union[int, float, list],
    val2: Union[int, float, list],
    aggregation_func=np.mean,
) -> float:
    """
    Calculate numeric similarity as positive difference between the values divided by their average. If lists are
        passed, uses `aggregation_func` (default np.mean) to convert the list of numbers into a single number.

    Args:
        val1 (Union[int, float, list])
        val2 (Union[int, float, list])
        aggregation_func (optional): function to convert list into numeric values if list is passed to either `val1`
            or `val2`.

    Returns:
        float: 0 <= f <= 1
    """

    if (isinstance(val1, (str, list)) and (len(val1) == 0)) or (
        isinstance(val2, (str, list)) and (len(val2) == 0)
    ):
        return 0

    try:
        val1 = (
            aggregation_func([float(v) for v in val1])
            if isinstance(val1, list)
            else float(val1)
        )

        val2 = (
            aggregation_func([float(v) for v in val2])
            if isinstance(val2, list)
            else float(val2)
        )
    except ValueError:
        # if either value can't be converted to a float, return 0 similarity
        logger.warning(
            f"Numeric similarity failed for values {val1} and {val2}. 0 similarity returned."
        )
        return 0

    diff = 1 - (abs(val1 - val2) / np.max([abs(val1), abs(val2)]))

    if diff < 0:
        return 0
    elif diff > 1:
        return 1
    else:
        return diff


def similarity_categorical(
    val1: Union[list, tuple, int, float, str],
    val2: Union[list, tuple, int, float, str],
    raise_on_diff_types=True,
) -> float:
    """
    Returns binary score of whether two items match. If lists are passed, returns positive (=1) if any item in List1
        is the same as any item in List2.

    Args:
        val1 (Union[list, tuple, int, float, str]): item or list of items
        val2 (Union[list, tuple, int, float, str]): item or list of items
        raise_on_diff_types (bool): whether to raise a ValueError if one of val1 is list-like and the other is a single value.
            If False, treats the single-value val as a single-element list, e.g. 'apple' -> ['apple'].

    Returns:
        float: 0 <= f <= 1
    """

    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
        intersection_size = len(set(val1).intersection(set(val2)))

        return 1 * (intersection_size > 0)

    elif isinstance(val1, (int, float, str)) and isinstance(val2, (int, float, str)):
        # True -> 1, False -> 0
        return 1 * (val1 == val2)

    else:
        # one of val1/val2 is list-like and the other is not

        if raise_on_diff_types:
            raise ValueError(
                """One of the provided values is list-like and the other is not. 
                To calculate similarity as if both values are lists set raise_on_diff_types=False."""
            )

        else:
            # converts int/float/str to one-element
            val_a1 = [i for i in [val1, val2] if isinstance(i, (int, float, str))]
            val_a2 = [i for i in [val1, val2] if isinstance(i, (list, tuple))][0]

            intersection_size = len(set(val_a1).intersection(set(val_a2)))

            return 1 * (intersection_size > 0)
