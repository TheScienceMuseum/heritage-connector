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
    time.sleep(2)
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        return sparql.query().convert()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("429 code : sleeping for 60 seconds")
            time.sleep(60)
            return get_sparql_results(endpoint_url, query)
        raise
