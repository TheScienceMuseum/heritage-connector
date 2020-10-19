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

    def add_score_to_search_results_df(
        self, res_df: pd.DataFrame, rank_col: str
    ) -> pd.DataFrame:
        """
        Add 'score' column to a dataframe using a column indicating the search result ranking of each item (starting at 1).
        Pass back the dataframe with the new column.
        """

        df = res_df.copy()

        max_rank = len(df)
        if max_rank == 1:
            df.at[0, "score"] = 1
        else:
            df["score"] = df["rank"].apply(lambda x: max_rank - x)
            sum_confidence = df["score"].sum()
            df["score"] = df["score"].apply(lambda x: x / sum_confidence)

        return df


class Classifier(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y) -> float:
        pass
