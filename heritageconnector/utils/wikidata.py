import requests
from typing import List, Union


class entities:
    def __init__(self, qcodes: Union[list, str], lang="en"):
        """
        One instance of this class per list of qcodes. The JSON response for a list of qcodes is made to Wikidata on 
        creation of a class instance. 

        Args:
            qcodes (str/list): Wikidata qcode or list of qcodes/
            lang (str, optional): [description]. Defaults to 'en'.
        """
        self.endpoint = (
            "http://www.wikidata.org/w/api.php?action=wbgetentities&format=json"
        )

        if isinstance(qcodes, str):
            qcodes = [qcodes]

        self.qcodes = qcodes
        self.properties = ["labels", "claims", "aliases"]
        self.lang = lang

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

        url = f"http://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={self._param_join(self.qcodes)}&props={self._param_join(self.properties)}&languages=en&languagefallback=1&formatversion=2"

        return requests.get(url).json()

    def get_labels(self) -> Union[list, str]:
        """
        Get label from Wikidata qcodes. Returns string if string is passed; list if list is passed.

        Returns:
            str/list: label or labels
        """

        labels = [
            self.response["entities"][qcode]["labels"][self.lang]["value"]
            for qcode in self.qcodes
        ]

        return labels if len(labels) > 1 else labels[0]

    def get_aliases(self) -> list:
        """
        Get aliases from Wikidata qcodes. Returns list if string is passed; list of lists if list is passed.

        Returns:
            list: of aliases
        """

        aliases = []

        for qcode in self.qcodes:
            response_aliases = self.response["entities"][qcode]["aliases"]
            if len(response_aliases) == 0:
                aliases.append([])
            else:
                aliases.append([item["value"] for item in response_aliases[self.lang]])

        return aliases if len(aliases) > 1 else aliases[0]

    def get_property_values(self, property_id) -> Union[str, list]:
        """
        Get the value or values for a given property ID.

        Args:
            property_id: ID to get the value of from a Wikidata record. 

        Returns:
            str/list: value or values of the specified property ID. 
        """

        property_vals = []

        for qcode in self.qcodes:
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

    def get_property_instance_of(self) -> Union[str, list]:
        """
        Gets the value of the 'instance of' property for each Wikidata item. 
        """

        return self.get_property_values("P31")
