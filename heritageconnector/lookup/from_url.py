from urllib.parse import urlparse
import urllib
import time
import json
import re
from ..utils.sparql import get_sparql_results
from ..utils.generic import extract_json_values

# methods to exchange URLs for IDs (e.g. wikidata ID)


class wikidata_id:
    """
    Get a Wikidata ID from a URL, where this process involves a direct lookup on Wikidata.

    Attributes:
        _domain_method_mapping (dict): mapping from domains in the format XXXX.co.uk to class methods
    """

    """
        Wikidata Property IDs
        =====================

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
        # TODO (KD): get this to work when no http://
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

    @classmethod
    def lookup_wikidata_id(self, pid: str, uid: str) -> str:
        """
        Lookup UID on Wikidata against given property ID of source

        Args:
            pid (str): Property ID of source (e.g. OxDnB ID: P1415)
            uid (str): Value of source ID (e.g. an OxDnB ID: 23105)

        Returns:
            qcode: Wikidata qcode in format Q(d+)
        """

        endpoint_url = "https://query.wikidata.org/sparql"

        query = f"""
            SELECT ?item ?itemLabel WHERE {{
                ?item wdt:{pid} "{uid}".
                SERVICE wikibase:label {{
                    bd:serviceParam wikibase:language "en" .
                }}
            }}
        """

        res = get_sparql_results(endpoint_url, query)

        if res:
            wikidata = res["results"]["bindings"]
            if wikidata and wikidata[0]:
                wikidata_url = wikidata[0]["item"]["value"]
                wikidata_id = re.findall(r"(Q\d+)", wikidata_url)[0]

                return wikidata_id

    @classmethod
    def from_oxdnb(self, url):
        """
        Given an Oxford DNB URL e.g. https://www.oxforddnb.com/view/article/23105, return the Wikidata ID.
        """

        uid = re.findall(r"/article/(\d+)", url)[0]
        qcode = self.lookup_wikidata_id("P1415", uid)

        return qcode

    @classmethod
    def from_wikipedia(self, url):
        """
        Given a Wikipedia URL e.g. https://en.wikipedia.org/wiki/Joseph_Lister, return the Wikidata ID.
        """

        path = re.findall("/wiki/(.*)", url)[0]
        endpoint = (
            "https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles="
            + path
            + "&format=json"
        )
        res = urllib.request.urlopen(endpoint)
        res_body = res.read()
        data = json.loads(res_body.decode("utf-8"))
        wikibase_item = extract_json_values(data, "wikibase_item")
        if wikibase_item and wikibase_item[0]:
            return wikibase_item[0]
        else:
            return
