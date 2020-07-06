from abc import ABC, abstractmethod
import pandas as pd


class TextSearch(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def run_search(self, text: str, **kwargs) -> pd.DataFrame:
        """
        Run a search on a text string, returning search results from the specified search engine. Results are returned as a dataframe with 
        a row for each result, and columns rank, item and itemLabel. 

        The kwargs below are recommended for filtering with Wikidata properties.

        Args:
            text (str): text to be searched

        Kwargs (wikidata):
            instanceof_filter (str): only return results that are an instance of this Wikidata property
            include_subclasses (bool): change the filter specified using `instanceof_filter` to look in the instance of/(subclass of)^n tree
            property_filters (dict): Wikidata properties and their specified values: {"P123": "Q312", ...}

        Returns:
            pd.DataFrame: with columns rank, item, itemLabel
        """

        pass
