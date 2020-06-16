from urllib.parse import urlsplit
import urllib
import time
import json
import re
from ..utils.sparql import get_sparql_results
from ..utils.generic import extract_json_values
import requests

# methods to exchange URLs for IDs (e.g. wikidata ID)


class wikidata_id:
    """
    Get a Wikidata ID from a URL, where this process involves a direct lookup on Wikidata.
    """

    """
        Wikidata Property IDs
        =====================

        oxfordnb.com => https://www.wikidata.org/wiki/Property:P1415
        getty.edu (ULAN) => https://www.wikidata.org/wiki/Property:P245
        gracesguide.co.uk => https://www.wikidata.org/wiki/Property:P3074
        books.google.co.uk => https://www.wikidata.org/wiki/Property:P675
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
            "en.wikipedia.org": self.from_wikipedia,  # we use the API
            "getty.edu": self.from_getty,  # there's more than 1 URL
            "britannica.com": self.from_britannica,  # old url formats in collection
        }

        # domain: (uid_from_url_regex, pid)
        # uid_regex is the regex in re.findall(uid_regex, url)[0]
        # that produces the URL
        self._domain_regex_mapping = {
            "oxforddnb.com": (r"/article/(\d+)", "P1415"),
            "gracesguide.co.uk": (r"gracesguide.co.uk/(.*)", "P3074"),
            "books.google.co.uk": (r"id=(\w+)", "P675"),
            "discovery.nationalarchives.gov.uk": (r"details/c/([A-Z]\d+)", "P3029"),
            "viaf.org": (r"/viaf/([1-9]\d(\d{0,7}|\d{17,20}))", "P214"),
        }

    @classmethod
    def get_enabled_domains(self):
        return tuple(self()._domain_method_mapping.keys()) + tuple(
            self()._domain_regex_mapping.keys()
        )

    @staticmethod
    def get_domain_from_url(url):
        """
        Gets the domain in the format XXXX.co.uk from the URL.

        Args:
            url (str): URL in the usual format.
        """

        try:
            # if the url doesn't start with http, the main bit of the url ends up in path
            domain = urlsplit(url).netloc or urlsplit(url).path

            # remove www. and slug if present
            domain = re.sub("^www.", "", domain)
            domain = re.sub("/.*$", "", domain)

            return domain

        except:
            raise Exception(f"PARSING FAILED: {url}")

    @classmethod
    def get(self, url: str) -> str:
        """
        Resolves URL to a Wikidata ID.
        """

        domain = self.get_domain_from_url(url)

        if domain in self()._domain_method_mapping.keys():
            return self()._domain_method_mapping[domain](url)

        elif domain in self()._domain_regex_mapping.keys():
            pattern, pid = self()._domain_regex_mapping[domain]
            return self.from_regex(url, pattern, pid)

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
    def from_regex(self, url: str, uid_pattern: str, pid: str) -> str:
        """
        Given an Oxford DNB URL e.g. https://www.oxforddnb.com/view/article/23105, return the Wikidata ID.

        Args:
            url (str): the URL of the page
            uid_pattern (str): the regex pattern to extract the uid from the URL
            pid (str): the Wikidata property ID for the domain
        Returns:
            qcode (str)
        """

        matches = re.search(uid_pattern, url)

        # if there's a match return the qcode, if not return empty string
        #  TODO: fail better
        if matches[1]:
            uid = matches[1]
            qcode = self.lookup_wikidata_id(pid, uid)
            return qcode
        else:
            return ""

    @classmethod
    def from_wikipedia(self, url: str) -> str:
        """
        Given a Wikipedia URL e.g. https://en.wikipedia.org/wiki/Joseph_Lister, return the Wikidata ID.

        Args:
            url (str)
        Returns: 
            qcode (str)
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

    @classmethod
    def from_getty(self, url: str) -> str:
        """
        Given a Getty URL e.g. 
        https://www.getty.edu/vow/ULANFullDisplay?find=Wheldon&role=&nation=&prev_page=1&subjectid=500044753 
        or http://vocab.getty.edu/page/ulan/500044753, return the Wikidata ID.

        Args:
            url (str)
        Returns: 
            qcode (str)
        """

        # extract ID from URL
        if "ulanfulldisplay" in url.lower():
            uid = re.findall(r"subjectid=(\d+)", url)[0]
        elif "ulan" in url.lower():
            uid = re.findall(r"/ulan/(\d+)", url)[0]

        return self.lookup_wikidata_id("P245", uid)

    @classmethod
    def from_britannica(self, url: str) -> str:
        """
        Given a britannica.com URL, return the Wikidata ID.

        Args:
            url (str)
        Returns: 
            qcode (str)
        """

        # get redirected (new) URL
        r = requests.get(url)
        redirected_url = r.url

        pattern = r"((?:biography|topic|place|science|animal|event|art|technology|plant|sports)\/.*)$"
        pid = "P1417"

        return self.from_regex(redirected_url, pattern, pid)
