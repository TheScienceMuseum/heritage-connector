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
