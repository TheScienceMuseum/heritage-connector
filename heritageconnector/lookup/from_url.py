# methods to exchange URLs for IDs (e.g. wikidata ID)


class wikidata_id:
    """
    Get a Wikidata ID from a URL, where this process involves a direct lookup on Wikidata. 

    Attributes:
        _domain_method_mapping (dict): mapping from domains in the format XXXX.co.uk to class methods
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

        url = ""

        return url

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

    @staticmethod
    def from_oxdnb(url):
        """
        Given an Oxford DNB URL e.g. https://www.oxforddnb.com/view/article/23105, return the Wikidata ID.
        """

        return

    @staticmethod
    def from_wikipedia(url):
        """
        Given a Wikipedia URL e.g. https://en.wikipedia.org/wiki/Joseph_Lister, return the WIkidata ID.
        """

        return
