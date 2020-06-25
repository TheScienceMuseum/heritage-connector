# All methods for calling sparql databases

from SPARQLWrapper import SPARQLWrapper, JSON
import urllib
import time


def get_sparql_results(endpoint_url: str, query: str) -> dict:
    """
    Makes a SPARQL query to endpoint_url.

    Args:
        endpoint_url (str): query endpoint
        query (str): SPARQL query

    Returns:
        query_result (dict): the JSON result of the query as a dict
    """
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setMethod("POST")
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader(
        "User-Agent",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36",
    )
    try:
        return sparql.query().convert()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            if "retry-after" in e.headers:
                if isinstance(e.headers["retry-after"], int):
                    time.sleep(e.headers["retry-after"])
            else:
                time.sleep(10)
            return get_sparql_results(endpoint_url, query)
        raise
