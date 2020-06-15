from urllib.parse import urlparse, request
from qwikidata.sparql import return_sparql_query_results
import json
import re

# methods to exchange URLs for IDs (e.g. wikidata ID)


class wikidata_id:
    """
    Get a Wikidata ID from a URL, where this process involves a direct lookup on Wikidata.

    Attributes:
        _domain_method_mapping (dict): mapping from domains in the format XXXX.co.uk to class methods
    """

    """
        Wikidata IDs
        ============

        oxfordnb.com => https://www.wikidata.org/wiki/Property:P1415
        getty.edu => https://www.wikidata.org/wiki/Property:P1667
        graceguide.co.uk => https://www.wikidata.org/wiki/Property:P3074
        britannica.com => https://www.wikidata.org/wiki/Property:P1417
        nationalarchives.gov.uk => https://www.wikidata.org/wiki/Property:P3029
        npg.org.uk => https://www.wikidata.org/wiki/Property:P1816
        jsror.org => https://www.wikidata.org/wiki/Q19832971
        archive.org => https://www.wikidata.org/wiki/Property:P724
        viaf.org => https://www.wikidata.org/wiki/Property:P214
        id.loc.gov => https://www.wikidata.org/wiki/Property:P244

        oxfordreference.com
        camerapedia
        radiomuseum.org.uk
        adlerplkanetarium.org
        steamindex.com
    """

    def __init__(self):
        self._domain_method_mapping = {
            "oxforddnb.com": self.from_oxdnb,
            "getty.edu": self.from_getty,
            "en.wikipedia.org": self.from_wikipedia,
        }

    @property
    def enabled_domains(self):
        return self._enabled_domains

    @staticmethod
    def get_domain_from_url(url):
        """
        Gets the domain in the format XXXX.co.uk from the URL.

        Args:
            url (str): URL in the usual format.
        """

        domain = urlparse(url).netloc

        return domain

    def check_domain_enabled(self, domain):
        """
        Checks whether the domain can be handled.

        Args:
            domain (str): in the format XXXX.co.uk

        Returns:
            boolean
        """

        return domain in self._domain_method_mapping.keys()

    def get(self, url):
        """
        Resolves URL to a Wikidata ID.
        """

        domain = self.get_domain_from_url(url)

        if self.check_domain_enabled(domain):
            return self._domain_method_mapping[domain](url)

    def lookup_wikidata_id(pid, uid):
        """
        lookup UID on Wikidata against given property ID of source
        """

        sparql_query = f"""
            SELECT ?item ?itemLabel WHERE {{
                ?item wdt:"{pid}" "{uid}".
                SERVICE wikibase:label {{
                    bd:serviceParam wikibase:language "en" .
                }}
            }}
        """

        res = return_sparql_query_results(sparql_query)
        qcode = res["results"]["bindings"][0]["item"]["value"]

        return qcode

    def from_oxdnb(self, url):
        """
        Given an Oxford DNB URL e.g. https://www.oxforddnb.com/view/article/23105, return the Wikidata ID.
        """

        match = re.search(r"/article/(\d+)", url)
        uid = match[1]
        qcode = self.lookup_wikidata_id("P1415", uid)

        return qcode

    def from_wikipedia(self, url):
        """
        Given a Wikipedia URL e.g. https://en.wikipedia.org/wiki/Joseph_Lister, return the WIkidata ID.
        """

        path = urlparse(url).path
        endpoint = (
            "https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles="
            + path
            + "&format=json"
        )
        res = request.urlopen(endpoint)
        res_body = res.read()
        data = json.loads(res_body.decode("utf-8"))
        wikibase_item = self.extract_json_values(data, "wikibase_item")
        if wikibase_item and wikibase_item[0]:
            return wikibase_item[0]
        else:
            return

    @staticmethod
    def extract_json_values(obj, key):
        """ Pull all values of specified key from nested JSON."""
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
