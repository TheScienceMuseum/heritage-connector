"""
Utilities for helping preprocess data before importing into Heritage Connector. 
For examples see `smg_jobs/smg_loader.py`.
"""

from heritageconnector.entity_matching.lookup import DenonymConverter

denonym_converter = DenonymConverter()


def get_country_from_nationality(nationality):
    """
    Uses list of denonyms to convert nationalities ("British") into countries ("United Kingdom").
    """
    country = denonym_converter.get_country_from_nationality(nationality)

    if country is not None:
        return country
    else:
        return nationality


def process_text(text: str) -> str:
    """
    Remove newline and tab characters and convert to string.
    """
    newstr = str(text)
    newstr = newstr.replace("\n", " ")
    newstr = newstr.replace("\t", " ")

    return newstr


def split_list_string(l: list, convert_to_lowercase=True):
    """
    Splits string separated by either commas or semicolons into a list. By default the items in this list are lowercased according to the 
    `convert_to_lowercase` argument. Each item in the list has no leading or trailing whitespace.

    Example:
    ```
    assert split_list_string("engineer, mathematical instrument maker ; lawyer") == ["engineer", "mathematical instrument maker", "lawyer"]
    ```
    """

    if convert_to_lowercase:
        return [
            x.strip().lower()
            for x in str(l).replace(";", ",").split(",")
            if x.strip() != ""
        ]
    else:
        return [
            x.strip() for x in str(l).replace(";", ",").split(",") if x.strip() != ""
        ]
