import requests
from typing import List, Union
from tqdm import tqdm
import re


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
