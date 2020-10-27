# All methods for calling sparql databases

from SPARQLWrapper import SPARQLWrapper, JSON
import urllib
import time
import json
import sys
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from heritageconnector.config import config
from heritageconnector import logging

logger = logging.get_logger(__name__)


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def get_sparql_results(endpoint_url: str, query: str) -> dict:
    """
    Makes a SPARQL query to endpoint_url.

    Args:
        endpoint_url (str): query endpoint
        query (str): SPARQL query

    Returns:
        query_result (dict): the JSON result of the query as a dict
    """
    user_agent = generate_user_agent()

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
