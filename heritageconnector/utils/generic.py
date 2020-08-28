# General utils for heritageconnector that don't fit anywhere else.
import requests


def extract_json_values(obj: dict, key: str) -> list:
    """
    Pull all values of specified key from nested JSON.

    Args:
        obj (dict): nested dict
        key (str): name of key to pull out

    Returns:
        list: [description]
    """
    arr = []

    def extract(obj, arr, key):
        """ Recursively search for values of key in JSON tree. """
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results


def get_redirected_url(url: str) -> str:
    """
    Given a URL, return the URL it redirects to.

    Args:
        url (str)
    """

    r = requests.get(url)
    return r.url


def add_dicts(dict1, dict2) -> dict:
    """
    Return a dictionary with the sum of the values for each key in both dicts. 
    """
    return {x: dict1.get(x, 0) + dict2.get(x, 0) for x in set(dict1).union(dict2)}


def flatten_list_of_lists(l: list) -> list:
    """
    [[1, 2], [3]] -> [1, 2, 3]
    """

    return [item for sublist in l for item in sublist]
