import requests
from typing import List, Set, Union
from tqdm import tqdm
import re
import os
from itertools import product
from heritageconnector.config import config
from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.utils.generic import cache
from heritageconnector import logging

logger = logging.get_logger(__name__)


class entities:
    def __init__(self, qcodes: Union[list, str], lang="en", page_limit=50):
        """
        One instance of this class per list of qcodes. The JSON response for a list of qcodes is made to Wikidata on 
        creation of a class instance. 

        Args:
            qcodes (str/list): Wikidata qcode or list of qcodes/
            lang (str, optional): Defaults to 'en'.
            page_limit (int): page limit for Wikidata API. Usually 50, can reach 500. 
        """
        self.endpoint = (
            "http://www.wikidata.org/w/api.php?action=wbgetentities&format=json"
        )

        if isinstance(qcodes, str):
            qcodes = [qcodes]

        self.qcodes = qcodes
        self.properties = ["labels", "claims", "aliases"]
        self.lang = lang
        self.page_limit = page_limit

        # get json response
        self.response = self.get_json()

    @staticmethod
    def _param_join(params: List[str]) -> str:
        """
        Joins list of parameters for the URL. ['a', 'b'] -> "a%7Cb"

        Args:
            params (list): list of parameters (strings)

        Returns:
            str
        """

        return "%7C".join(params) if len(params) > 1 else params[0]

    def get_json(self) -> dict:
        """
        Get json response through the `wbgetentities` API.

        Returns:
            dict: raw JSON response from API
        """

        qcodes_paginated = [
            self.qcodes[i : i + self.page_limit]
            for i in range(0, len(self.qcodes), self.page_limit)
        ]
        all_responses = {}
        logger.info(
            f"Getting {len(self.qcodes)} wikidata documents in pages of {self.page_limit}"
        )

        for page in tqdm(qcodes_paginated):
            url = f"http://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={self._param_join(page)}&props={self._param_join(self.properties)}&languages=en&languagefallback=1&formatversion=2"
            response = requests.get(url).json()
            all_responses.update(response["entities"])

        return {"entities": all_responses}

    def get_labels(self, qcodes=None) -> Union[list, str]:
        """
        Get label from Wikidata qcodes. Returns string if string is passed; list if list is passed.

        Args: 
            qcodes (list, optional): subset of all qcodes to pass in

        Returns:
            str/list: label or labels
        """

        if qcodes is not None:
            assert all(elem in self.qcodes for elem in self.qcodes)
        else:
            qcodes = self.qcodes

        labels = []
        for qcode in qcodes:
            try:
                labels.append(
                    self.response["entities"][qcode]["labels"][self.lang]["value"]
                )
            except KeyError:
                # if there is no value in the correct language, return an empty string
                labels.append("")

        return labels if len(labels) > 1 else labels[0]

    def get_aliases(self, qcodes=None) -> list:
        """
        Get aliases from Wikidata qcodes. Returns list if string is passed; list of lists if list is passed.

        Args: 
            qcodes (list, optional): subset of all qcodes to pass in

        Returns:
            list: of aliases
        """

        if qcodes is not None:
            assert all(elem in self.qcodes for elem in self.qcodes)
        else:
            qcodes = self.qcodes

        aliases = []

        for qcode in qcodes:
            response_aliases = self.response["entities"][qcode]["aliases"]
            if len(response_aliases) == 0:
                aliases.append([])
            else:
                aliases.append([item["value"] for item in response_aliases[self.lang]])

        return aliases if len(aliases) > 1 else aliases[0]

    def get_property_values(self, property_id, qcodes=None) -> Union[str, list]:
        """
        Get the value or values for a given property ID.

        Args:
            property_id: ID to get the value of from a Wikidata record. 

        Returns:
            str/list: value or values of the specified property ID. 
        """

        if qcodes is not None:
            assert all(elem in self.qcodes for elem in self.qcodes)
        else:
            qcodes = self.qcodes

        property_vals = []

        for qcode in qcodes:
            try:
                property_data = self.response["entities"][qcode]["claims"][property_id]
            except KeyError:
                raise KeyError(
                    f"Property {property_id} does not exist for item {qcode}. Check the Wikidata page or try another property ID."
                )

            qcode_vals = []
            for item in property_data:
                qcode_vals.append(item["mainsnak"]["datavalue"]["value"]["id"])

            property_vals.append(qcode_vals)

        return property_vals if len(property_vals) > 1 else property_vals[0]

    def get_property_instance_of(self, qcodes=None) -> Union[str, list]:
        """
        Gets the value of the 'instance of' property for each Wikidata item. 
        """

        return self.get_property_values("P31", qcodes)


@cache(os.path.join(os.path.dirname(__file__), "../entitydistance.cache"))
def get_distance_between_entities_cached(
    qcode_set: Set[str], reciprocal: bool = False, max_path_length: int = 10,
) -> float:
    res = get_distance_between_entities(qcode_set, reciprocal, max_path_length)

    return res


def get_distance_between_entities(
    qcode_set: Set[str], reciprocal: bool = False, max_path_length: int = 10,
) -> float:
    """
    Get the length of the shortest path between two entities in `qcode_set`along the 'subclass of' axis. Flag `reciprocal=True` 
    returns 1/(1+l) where l is the length of the shortest path, which can be treated as a similarity measure.

    Args:
        qcode_set (Set[str])
        reciprocal (bool, optional): Return 1/(1+l), where l is the length of the shortest path. Defaults to False.
        max_iterations (int, optional): Maximum iterations to look for the shortest path. If the actual shortest path is  
            greater than max_iterations, 10*max_iterations (reciprocal=False) or 1/(1+10*max_iterations) (reciprocal=True) is returned.

    Returns:
        float: distance (d <= max_iterations or max_iterations*10) or reciprocal distance (0 < d <= 1)
    """

    if len(qcode_set) == 1:
        # identity - assume two values have been passed in even though the set will have length 1
        return 1 if reciprocal else 0

    if len(qcode_set) != 2:
        raise ValueError("Input variable qcode_set must contain exactly 1 or 2 items")

    qcodes = [i for i in qcode_set]

    if (qcodes[0] == "") or (qcodes[1] == ""):
        # at least one value is empty so return maximum dissimilarity
        return 0 if reciprocal else 1
    else:
        raise_invalid_qid(qcodes[0])
        raise_invalid_qid(qcodes[1])

    link_type = "P279"

    query = f"""PREFIX gas: <http://www.bigdata.com/rdf/gas#>

    SELECT ?super (?aLength + ?bLength as ?length) WHERE {{
    SERVICE gas:service {{
        gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                    gas:in wd:{qcodes[0]} ;
                    gas:traversalDirection "Forward" ;
                    gas:out ?super ;
                    gas:out1 ?aLength ;
                    gas:maxIterations {max_path_length} ;
                    gas:linkType wdt:{link_type} .
    }}
    SERVICE gas:service {{
        gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                    gas:in wd:{qcodes[1]} ;
                    gas:traversalDirection "Forward" ;
                    gas:out ?super ;
                    gas:out1 ?bLength ;
                    gas:maxIterations {max_path_length} ;
                    gas:linkType wdt:{link_type} .
    }}  
    }} ORDER BY ?length
    LIMIT 1
    """

    result = get_sparql_results(config.WIKIDATA_SPARQL_ENDPOINT, query)["results"][
        "bindings"
    ]

    if len(result) == 0:
        distance = 10 * max_path_length
    else:
        distance = int(float(result[0]["length"]["value"]))

    return 1 / (1 + distance) if reciprocal else distance


def get_distance_between_entities_multiple(
    qcode_set: Set[Union[str, tuple]],
    reciprocal: bool = False,
    max_path_length: int = 10,
    use_cache: bool = True,
) -> float:
    """
    Get the length of the shortest path between entities or sets of entities along the 'subclass of' axis. When two entities
    [a,b] are passed the shortest path is returned, and when two groups [(a,b), (c,d)] are passed the shortest path between
    the closest entities across those two groups is returned.

    Flag `reciprocal=True` returns 1/(1+l) where l is the length of the shortest path, which can be treated as a similarity measure.

    Args:
        qcode_set (Set[Union[str, list], Union[str, list]])
        reciprocal (bool, optional): Return 1/(1+l), where l is the length of the shortest path. Defaults to False.
        max_iterations (int, optional): Maximum iterations to look for the shortest path. If the actual shortest path is  
            greater than max_iterations, 10*max_iterations (reciprocal=False) or 1/(1+10*max_iterations) (reciprocal=True) is returned.
        use_cache (bool, optional): whether to use a query cache stored on disk. Defaults to True

    Returns:
        Union[float, int]: distance (int <= max_iterations or max_iterations*10) or reciprocal distance (float, 0 < f <= 1)
    """

    if len(qcode_set) == 1:
        # identity - assume two values have been passed in even though the set will have length 1
        return 1 if reciprocal else 0

    if len(qcode_set) != 2:
        raise ValueError("Input variable qcode_set must contain exactly 1 or 2 items")

    # Convert set into list so we can access its individual values.
    # (The result of the distance function is independent of order)
    qcode_list = [i for i in qcode_set]
    qcode_1, qcode_2 = qcode_list

    if (qcode_1 == "") or (qcode_2 == ""):
        # one value is empty so return maximum dissimilarity
        return 0 if reciprocal else 1

    if isinstance(qcode_1, list):
        qcode_1 = tuple(qcode_1)
    if isinstance(qcode_2, list):
        qcode_2 = tuple(qcode_2)

    if isinstance(qcode_1, str):
        raise_invalid_qid(qcode_1)
        qcode_1 = [qcode_1]
    elif isinstance(qcode_1, tuple) and len(qcode_1) > 0:
        [raise_invalid_qid(q) for q in qcode_1]
    else:
        raise ValueError(
            f"Item of qcode_set {qcode_1} is either not a string or is an empty list"
        )

    if isinstance(qcode_2, str):
        raise_invalid_qid(qcode_2)
        qcode_2 = [qcode_2]
    elif isinstance(qcode_2, tuple) and len(qcode_2) > 0:
        [raise_invalid_qid(q) for q in qcode_2]
    else:
        raise ValueError(
            f"Item of qcode_set {qcode_2} is either not a string or is an empty list"
        )

    combinations = product(list(set(qcode_1)), list(set(qcode_2)))
    result_list = []

    if use_cache:
        for q1, q2 in combinations:
            result_list.append(
                get_distance_between_entities_cached(
                    {q1, q2}, reciprocal=reciprocal, max_path_length=max_path_length
                )
            )
    else:
        for q1, q2 in combinations:
            result_list.append(
                get_distance_between_entities(
                    {q1, q2}, reciprocal=reciprocal, max_path_length=max_path_length
                )
            )

    if reciprocal:
        return max(result_list)
    else:
        return min(result_list)


def url_to_qid(url: Union[str, list], raise_invalid=True) -> Union[str, list]:
    """
    Maps Wikidata URL of an entity to QID e.g. http://www.wikidata.org/entity/Q7187777 -> Q7187777.

    Args:
        raise_invalid (bool, optional): whether to raise if a QID can't be found in the URL. If False, 
            for any string in which a QID can't be found an empty string is returned.
    """

    if isinstance(url, str):
        found = re.findall(r"(Q\d+)", url)
        if len(found) == 1:
            return found[0]
        else:
            if raise_invalid:
                raise ValueError("URL does not contain a single QID")
            else:
                return ""

    elif isinstance(url, list):
        return [url_to_qid(i, raise_invalid) for i in url]


def qid_to_url(qid: Union[str, list]) -> Union[str, list]:
    """
    Maps QID of an entity to a Wikidata URL e.g. Q7187777 -> http://www.wikidata.org/entity/Q7187777.
    """

    if isinstance(qid, str):
        return f"http://www.wikidata.org/entity/{qid}"
    elif isinstance(qid, list):
        return [qid_to_url(i) for i in qid]


def url_to_pid(url: Union[str, list]) -> Union[str, list]:
    """
    Maps Wikidata URL of an entity to PID e.g. http://www.wikidata.org/prop/direct/P570 -> P570.
    """

    if isinstance(url, str):
        return re.findall(r"(P\d+)", url)[0]
    elif isinstance(url, list):
        return [url_to_pid(i) for i in url]


def pid_to_url(pid: str) -> str:
    """
    Maps PID of an entity to a Wikidata URL e.g. P570 -> http://www.wikidata.org/prop/direct/P570.
    """

    if isinstance(pid, str):
        return f"http://www.wikidata.org/entity/{pid}"
    elif isinstance(pid, list):
        return [pid_to_url(i) for i in pid]


def raise_invalid_qid(qid: str) -> str:
    """
    Raise ValueError if supplied value is not a valid QID
    """

    if not isinstance(qid, str):
        raise ValueError(f"QID {qid} is not of type string")

    if len(re.findall(r"(Q\d+)", qid)) != 1:
        raise ValueError(f"QID {qid} is not a valid QID")


def join_qids_for_sparql_values_clause(qids: list) -> str:
    """
    Return joined list of QIDs for VALUES clause in a SPARQL query.
    E.g. VALUES ?item {wd:Q123 wd:Q234}

    Args:
        qids (list): list of QIDs

    Returns:
        str: QIDs formatted for VALUES clause
    """

    return " ".join([f"wd:{i}" for i in qids])


def filter_qids_in_class_tree(qids: list, higher_class: str) -> list:
    """
    Returns filtered list of QIDs that exist in the class tree below the QID defined by 
    `higher_class`. Raises if higher_class is not a valid QID.

    Args:
        qids (list): list of QIDs
        higher_class (str): QID of higher class

    Returns:
        list
    """

    formatted_qids = join_qids_for_sparql_values_clause(qids)

    # assume format of each item of qids has already been checked
    # TODO: what's a good pattern for coordinating this checking so it's not done multiple times?
    raise_invalid_qid(higher_class)

    query = f"""SELECT * WHERE {{
    VALUES ?item {{ {formatted_qids} }}
    ?item wdt:P279* wd:{higher_class}.
    }}"""

    res = get_sparql_results(config.WIKIDATA_SPARQL_ENDPOINT, query)

    return [url_to_qid(i["item"]["value"]) for i in res["results"]["bindings"]]
