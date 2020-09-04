from fuzzywuzzy import fuzz
from typing import Union
from itertools import product
import numpy as np

# Similarity measures to compare fields of different types
# Each function should use the following template:
# ```
# def similarity_<type>(val1, val2, **kwargs) -> float:
#   ...
#   # 0 <= sim <= 1
#   return sim
# ```

np.seterr(all="raise")


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
        return 0

    return 1 - (abs(val1 - val2) / np.mean([val1, val2]))


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
