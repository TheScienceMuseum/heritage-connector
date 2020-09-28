import pandas as pd
import re
from heritageconnector import logging

logger = logging.get_logger(__name__)
# methods to transform different typed fields


def transform_series_str_to_list(series: pd.Series, separator: str) -> pd.Series:
    """
    Splits the list column according to a specified separator. Removes leading and trailing whitespace
    from each list element and converts it to lowercase.

    Null values: returns empty list.

    Args:
        series (pd.Series)
        separator (str)

    Returns:
        pd.Series
    """

    return (
        series.fillna("")
        .astype(str)
        .apply(lambda i: [x.strip().lower() for x in i.split(separator)])
    )


def get_year_from_date_value(datestring: str, raise_errors: bool = False) -> int:
    """
    Looks for a year mention in a date-like string by finding a run of 1-4 digits if BCE, 
    or 4 digits if not BCE.

    Returns None if no date found, the date if only 1 is found, the average of the two if 
    two dates are found, and the first date if more than 2 dates are found.

    Args:
        date (str)

    Returns:
        str:
    """

    datestring = str(datestring)

    if "BCE" in datestring:
        datestring = datestring.replace("BCE", "").strip()
        year_matches = re.findall(r"(\d{1,4})", datestring)
        # BCE dates are recorded in Wikidata as negative years
        year_matches = [-1 * int(match) for match in year_matches]

    else:
        # look for (\d{4)) - avoiding trying to convert "about 1984ish" into
        # a date format using datetime
        year_matches = re.findall(r"(\d{4})", datestring)

    try:
        if len(year_matches) == 0:
            return None
        elif len(year_matches) == 1 or len(year_matches) > 2:
            return int(year_matches[0])
        elif len(year_matches) == 2:
            # assume in the format "333-345 BCE" / "1983-1984"
            return (int(year_matches[0]) + int(year_matches[1])) / 2
    except ValueError as e:
        if raise_errors:
            raise e
        else:
            logger.error(e)


def assert_qid_format(qid: str):
    """
    Asserts that a string is of a valid QID format. 
    Raises ValueError if not.
    """

    if not re.match(r"q\d+", qid.lower()):
        raise ValueError(f"{qid} is an invalid Wikidata QID.")
