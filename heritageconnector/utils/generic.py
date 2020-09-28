# General utils for heritageconnector that don't fit anywhere else.
import requests
from itertools import islice
import shelve
import functools


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


def paginate_generator(generator, page_size: int):
    """
    Returns an iterator that returns items from the provided generator grouped into `page_size`.
    If the size of the output from the original generator isn't an exact multiple of 
    `page_size`, the last list returned by the iterator will be of size less than `page_size`.

    Returns:
        iterator of lists
    """
    return iter(lambda: list(islice(generator, page_size)), [])


def _check_cache(cache_, key, func, args, kwargs):
    if key in cache_:
        # use cached results
        # print("using cached results")
        return cache_[key]
    else:
        # no cache results: call function
        # print("no cache results: calling function")
        result = func(*args, **kwargs)
        cache_[key] = result
        return result


def cache(filename: str):
    """
    Decorator to cache function calls to a pickle file in the location specified by filename.
    """

    def decorating_function(user_function):
        def wrapper(*args, **kwargs):
            args = tuple([frozenset(i) if isinstance(i, set) else i for i in args])
            args_key = str(hash(functools._make_key(args, kwargs, typed=False)))
            func_key = ".".join([user_function.__module__, user_function.__name__])
            key = func_key + args_key
            handle_name = "{}_handle".format(filename)
            if hasattr(cache, handle_name) and not hasattr(
                getattr(cache, handle_name).dict, "closed"
            ):
                # print("Using open handle")
                return _check_cache(
                    getattr(cache, handle_name), key, user_function, args, kwargs
                )
            else:
                # print("Opening handle")
                with shelve.open(filename, writeback=True) as c:
                    setattr(
                        cache, handle_name, c
                    )  # Save a reference to the open handle
                    return _check_cache(c, key, user_function, args, kwargs)

        return functools.update_wrapper(wrapper, user_function)

    return decorating_function
