# All methods for calling sparql databases

from SPARQLWrapper import SPARQLWrapper, JSON
import urllib
import time
import json
import sys
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from heritageconnector.config import config
from heritageconnector import logging, namespace

logger = logging.get_logger(__name__)


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def get_sparql_results(endpoint_url: str, query: str, add_prefixes=True) -> dict:
    """
    Makes a SPARQL query to endpoint_url.

    Args:
        endpoint_url (str): query endpoint
        query (str): SPARQL query
        add_prefixes (bool, optional): whether to add PREFIX lines using the namespaces in heritageconnector.namespace. 
            Defaults to True.

    Returns:
        query_result (dict): the JSON result of the query as a dict
    """
    user_agent = generate_user_agent()

    if add_prefixes:
        prefix_text = generate_sparql_prefixes_header()
        query = prefix_text + " \n" + query

    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setMethod("POST")
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader(
        "User-Agent", user_agent,
    )
    try:
        return sparql.query().convert()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            logger.debug("429")
            if e.headers.get("retry-after", None):
                logger.debug(f"Retrying after {e.headers['retry-after']} seconds")
                time.sleep(int(e.headers["retry-after"]))
            else:
                time.sleep(10)
            return get_sparql_results(endpoint_url, query)
        elif e.code == 403:
            logger.debug("403")
            return e.read().decode("utf8", "ignore")
        raise e
    except json.decoder.JSONDecodeError as e:
        logger.error("JSONDecodeError. Query:")
        logger.error(query)
        raise e


def generate_user_agent() -> str:
    """
    Generates a User Agent header string according to the Wikidata policy
        (https://meta.wikimedia.org/wiki/User-Agent_policy)

    Returns:
        str: [description]
    """

    part_hc = "Heritage Connector bot/0.1"
    part_python = "Python/" + ".".join(str(i) for i in sys.version_info)
    part_requests = "requests/" + requests.__version__

    if "CUSTOM_USER_AGENT" in config.__dict__:
        return f"{part_hc} {part_requests} {part_python} ({config.CUSTOM_USER_AGENT})"
    else:
        return f"{part_hc} {part_requests} {part_python}"


def generate_sparql_prefixes_header() -> str:
    """
    Generate the header for SPARQL queries containing all the namespaces in `heritageconnector.namespace`.

    E.g.:
    ```
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> 
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    ...
    ```

    Returns:
        str: prefixes for top of SPARQL query
    """

    rdf_names = [
        i
        for i in namespace.__dict__.keys()
        if (not i.startswith("__")) and i != "Namespace"
    ]

    prefix_header = ""

    for name in rdf_names:
        prefix_header += f"PREFIX {name.lower()}: <{str(namespace.__dict__[name])}>\n"

    return prefix_header
