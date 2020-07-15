import pandas as pd

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
