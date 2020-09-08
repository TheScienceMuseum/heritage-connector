# All methods for calling sparql databases

from SPARQLWrapper import SPARQLWrapper, JSON
import urllib
import time
import json


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
        "Heritage Connector bot/0.1 (heritageconnector@gmail.com / Science Museum Group)",
    )
    try:
        return sparql.query().convert()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            if isinstance(e.headers.get("retry-after", None), int):
                time.sleep(e.headers["retry-after"])
            else:
                time.sleep(10)
            return get_sparql_results(endpoint_url, query)
        raise
    except json.decoder.JSONDecodeError as e:
        print(query)
        raise e
