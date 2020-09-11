import requests
from typing import List, Union
from tqdm import tqdm
import re
from heritageconnector.config import config
from heritageconnector.utils.sparql import get_sparql_results


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
        print(
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


def get_distance_between_entities(
    qcode_1: str, qcode_2: str, reciprocal=False, max_path_length=10
) -> Union[float, int]:
    """
    Get the length of the shortest path between two entities along the 'subclass of' axis. Flag `reciprocal=True`
        returns 1/(1+l), where l is the length of the shortest path. 

    Args:
        qcode_1 (str)
        qcode_2 (str)
        reciprocal (bool, optional): Return 1/(1+l), where l is the length of the shortest path. Defaults to False.
        max_iterations (int, optional): Maximum iterations to look for the shortest path. If the actual shortest path is  
            greater than max_iterations, 10*max_iterations (reciprocal=False) or 0 (reciprocal=True) is returned.

    Returns:
        Union[float, int]: distance (int <= max_iterations or max_iterations*10) or reciprocal distance (float, 0 < f <= 1)
    """

    raise_invalid_qid(qcode_1)
    raise_invalid_qid(qcode_2)

    link_type = "P279"

    query = f"""PREFIX gas: <http://www.bigdata.com/rdf/gas#>

    SELECT ?super (?aLength + ?bLength as ?length) WHERE {{
    SERVICE gas:service {{
        gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                    gas:in wd:{qcode_1} ;
                    gas:traversalDirection "Forward" ;
                    gas:out ?super ;
                    gas:out1 ?aLength ;
                    gas:maxIterations {max_path_length} ;
                    gas:linkType wdt:{link_type} .
    }}
    SERVICE gas:service {{
        gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                    gas:in wd:{qcode_2} ;
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
        # distance is greater than max_iterations
        return 0 if reciprocal else 10 * max_path_length
    else:
        # distance is within max_iterations, so actual distance is returned
        distance = int(float(result[0]["length"]["value"]))

        if reciprocal:
            return 1 / (1 + distance)
        else:
            return distance


def url_to_qid(url: str) -> str:
    """
    Maps Wikidata URL of an entity to QID e.g. http://www.wikidata.org/entity/Q7187777 -> Q7187777.
    """

    return re.findall(r"(Q\d+)", url)[0]


def qid_to_url(qid: str) -> str:
    """
    Maps QID of an entity to a Wikidata URL e.g. Q7187777 -> http://www.wikidata.org/entity/Q7187777.
    """

    return f"http://www.wikidata.org/entity/{qid}"


def url_to_pid(url: str) -> str:
    """
    Maps Wikidata URL of an entity to PID e.g. http://www.wikidata.org/prop/direct/P570 -> P570.
    """

    return re.findall(r"(P\d+)", url)[0]


def pid_to_url(pid: str) -> str:
    """
    Maps PID of an entity to a Wikidata URL e.g. P570 -> http://www.wikidata.org/prop/direct/P570.
    """

    return f"http://www.wikidata.org/prop/direct/{pid}"


def raise_invalid_qid(qid: str) -> str:
    """
    Raise ValueError if supplied value is not a valid QID
    """

    if not isinstance(qid, str):
        raise ValueError(f"QID {qid} is not of type string")

    if len(re.findall(r"(Q\d+)", qid)) != 1:
        raise ValueError(f"QID {qid} is not a valid QID")
