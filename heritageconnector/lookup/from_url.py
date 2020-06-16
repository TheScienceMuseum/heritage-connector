from urllib.parse import urlsplit
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
        getty.edu (artist names) => https://www.wikidata.org/wiki/Property:P245
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
        # TODO (KD): change to {domain: (method, uid_pattern, pid)}
        self._domain_method_mapping = {
            "oxforddnb.com": self.from_oxdnb,
            "en.wikipedia.org": self.from_wikipedia,
            "getty.edu": self.from_getty,
            "gracesguide.co.uk": self.from_graces_guide,
            "books.google.co.uk": self.from_google_books,
        }

    @classmethod
    def get_enabled_domains(self):
        return tuple(self()._domain_method_mapping.keys())

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
    def check_domain_enabled(self, domain):
        """
        Checks whether the domain can be handled.

        Args:
            domain (str): in the format XXXX.co.uk

        Returns:
            boolean
        """

        if domain in self.get_enabled_domains():
            return True
        else:
            raise ValueError(f"Domain {domain} not currently handled.")

    @classmethod
    def get(self, url):
        """
        Resolves URL to a Wikidata ID.
        """

        domain = self.get_domain_from_url(url)

        if self.check_domain_enabled(domain):
            return self()._domain_method_mapping[domain](url)

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
    def from_oxdnb(self, url: str):
        """
        Given an Oxford DNB URL e.g. https://www.oxforddnb.com/view/article/23105, return the Wikidata ID.
        """

        uid = re.findall(r"/article/(\d+)", url)[0]
        qcode = self.lookup_wikidata_id("P1415", uid)

        return qcode

    @classmethod
    def from_wikipedia(self, url: str):
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

    @classmethod
    def from_getty(self, url: str):
        """
        Given a Getty URL e.g. 
        https://www.getty.edu/vow/ULANFullDisplay?find=Wheldon&role=&nation=&prev_page=1&subjectid=500044753 
        or http://vocab.getty.edu/page/ulan/500044753, return the Wikidata ID.

        Args:
            url (str)
        """

        # extract ID from URL
        if "ulanfulldisplay" in url.lower():
            uid = re.findall(r"subjectid=(\d+)", url)[0]
        elif "ulan" in url.lower():
            uid = re.findall(r"/ulan/(\d+)", url)[0]

        return self.lookup_wikidata_id("P245", uid)

    @classmethod
    def from_graces_guide(self, url: str):
        """
        Given a Grace's Guide URL e.g. https://www.gracesguide.co.uk/Maudslay,_Sons_and_Field,
        return the Wikidata ID.

        Args:
            url (str)
        """

        uid = re.findall(r"gracesguide.co.uk/(.*)", url)[0]

        return self.lookup_wikidata_id("P3074", uid)

    @classmethod
    def from_google_books(self, url: str):
        """
        Given a Google Books URL e.g. https://books.google.co.uk/books?id=RMMRRho44EsC&hl=en,
        return the Wikidata ID.

        Args:
            url (str)
        """

        uid = re.findall(r"id=(\w+)")[0]

        return self.lookup_wikidata_id("P675", uid)
