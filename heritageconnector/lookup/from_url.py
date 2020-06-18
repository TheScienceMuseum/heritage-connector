from urllib.parse import urlsplit
import urllib
import time
import json
import re
from typing import Union
from ..utils.sparql import get_sparql_results
from ..utils.generic import extract_json_values, get_redirected_url

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

    def __init__(self, custom_patterns=None):
        """
        Args:
            custom_patterns (list(tuple), optional): Any custom regex patterns to extract uids from in `wikidata_id.get_from_free_text`, in the format [(r"pattern", pid)] where pid is the Wikidata Property ID.
        """
        self._domain_method_mapping = {
            "en.wikipedia.org": self.from_wikipedia,  # we use the API
            "getty.edu": self.from_getty,  # there's more than 1 URL
            "britannica.com": self.from_britannica,  # old url formats in collection
        }

        # domain: (uid_from_regex, pid)
        # uid_regex is the regex in re.findall(uid_regex, url)[0]
        # that produces the URL
        self._domain_regex_mapping = {
            "oxforddnb.com": (r"/article/(\d+)", "P1415"),
            "gracesguide.co.uk": (r"gracesguide.co.uk/(.*)", "P3074"),
            "books.google.co.uk": (r"id=(\w+)", "P675"),
            "discovery.nationalarchives.gov.uk": (r"details/c/([A-Z]\d+)", "P3029"),
            "viaf.org": (r"/viaf/([1-9]\d(\d{0,7}|\d{17,20}))", "P214"),
        }

        self.custom_patterns = custom_patterns
        self._check_custom_patterns()

    def _check_custom_patterns(self):
        """
        Assert that custom patterns is in the format list(list/tuple(str, str), ...).

        Raises:
            ValueError: Provides guidance on correct type of custom_patterns.
        """

        if self.custom_patterns:
            if (
                not isinstance(self.custom_patterns, (list))
                and all(
                    isinstance(item, (list, tuple)) for item in self.custom_patterns
                )
                and all(
                    isinstance(text, str)
                    for item in self.custom_patterns
                    for text in item
                )
            ):
                raise ValueError(
                    "Input variable custom_patterns must be follow the format lst(lst(str,str)), where lst is list/tuple"
                )

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

        except Exception:
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

        else:
            raise ValueError("URL not handled")

    def get_from_free_text(self, text: str, return_urls=False) -> list:
        """
        Gets all references to URLs and returns Wikidata IDs from any that can be parsed.

        Args:
            text (str): the text from which to extract the Wikidata IDs
            return_urls (bool): whether to return a list of URLs

        Returns:
            list: the Wikipedia IDs of each URL matched
            list (optional): the URLs extracted from the text
        """

        # find all URLs
        url_pattern = (
            r"((?:https?://|www\.|https?://|www\.)[a-z0-9\.:].*?(?=[\s;,!:\[\]]|$))"
        )
        urls = re.findall(url_pattern, text)

        # map all URLs to qcodes
        qcodes = []
        for url in urls:
            try:
                qcodes.append(self.get(url))
            except ValueError:
                pass

        # remove duplicates
        qcodes = list(set(qcodes))

        # get qcodes from uids in user-defined patterns
        if self.custom_patterns:
            for pattern, pid in self.custom_patterns:
                new_qcodes = self.from_regex(text, pattern, pid)

                if isinstance(new_qcodes, str) and len(new_qcodes) > 0:
                    qcodes.append(new_qcodes)
                elif isinstance(new_qcodes, list):
                    qcodes += new_qcodes

        # return lists
        return qcodes, urls if return_urls else qcodes

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
    def from_regex(self, text: str, uid_pattern: str, pid: str) -> Union[str, list]:
        """
        Given a string, PID and a regex pattern to extract the UID, return the Wikidata ID.

        Args:
            text (str): input text to extract the uid from
            uid_pattern (str): regex pattern to extract the uid from the text
            pid (str): the Wikidata property ID for the domain
        Returns:
            str if 0-1 outputs; list(str) if multiple
        """

        matches = re.findall(uid_pattern, text)

        qcodes = []
        for uid_match in matches:
            # if findall has returned more than one group, choose the longest group
            if isinstance(uid_match, tuple):
                uid_match = max(uid_match, key=len)
            qcodes.append(self.lookup_wikidata_id(pid, uid_match))

        if len(qcodes) == 0:
            return ""
        elif len(qcodes) == 1:
            return qcodes[0]
        else:
            return qcodes

    @classmethod
    def from_wikipedia(self, url: str) -> str:
        """
        Given a Wikipedia URL e.g. https://en.wikipedia.org/wiki/Joseph_Lister, return the Wikidata ID.

        Args:
            url (str)
        Returns: 
            qcode (str)
        """

        matches = re.findall("/wiki/(.*)", url)
        if len(matches) == 1:
            path = matches[0]

            # passing the redirects param through the API gets the details of the page that Wikipedia may redirect to
            endpoint = (
                "https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles="
                + path
                + "&format=json&redirects"
            )
            res = urllib.request.urlopen(endpoint)
            res_body = res.read()
            data = json.loads(res_body.decode("utf-8"))
            wikibase_item = extract_json_values(data, "wikibase_item")
            if len(wikibase_item) >= 1:
                return wikibase_item[0]
        else:
            return ""

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
            match = re.findall(r"subjectid=(\d+)", url)
        elif "ulan" in url.lower():
            match = re.findall(r"/ulan/(\d+)", url)
        else:
            match = []

        if len(match) == 1:
            uid = match[0]
            return self.lookup_wikidata_id("P245", uid)
        else:
            return ""

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
        redirected_url = get_redirected_url(url)

        pattern = r"((?:biography|topic|place|science|animal|event|art|technology|plant|sports)\/.*)$"
        pid = "P1417"

        return self.from_regex(redirected_url, pattern, pid)
