from heritageconnector.nlp.string_pairs import fuzzy_match
from fuzzywuzzy import fuzz
from heritageconnector.base.disambiguation import TextSearch
from heritageconnector.config import config
from heritageconnector.utils.sparql import get_sparql_results
import pandas as pd
import re


class wikidata_text_search(TextSearch):
    def __init__(self):
        super().__init__()

    def run_search(self, text: str, limit=100, **kwargs) -> pd.DataFrame:
        """
        Run Wikidata search.

        Args:
            text (str): text to search
            limit (int, optional): Defaults to 100.

        Kwargs:
            instanceof_filter (str/list): property or properties to filter values by instance of. 
            include_class_tree (bool): whether to look in the subclass tree for the instance of filter.
            property_filters (dict): filters on exact values of properties you want to pass through. {property: value, ...}

        Returns:
            pd.DataFrame: columns rank, item, itemLabel, score
        """

        class_tree = "/wdt:P279*" if "include_class_tree" in kwargs else ""
        sparq_property_filter = ""

        if "instanceof_filter" in kwargs:
            property_id = kwargs["instanceof_filter"]
            if isinstance(property_id, str):
                # one instanceof in filter
                sparq_instanceof = f"?item wdt:P31{class_tree} wd:{property_id}."
            elif isinstance(property_id, list):
                if len(property_id) == 1:
                    sparq_instanceof = f"?item wdt:P31{class_tree} wd:{property_id[0]}."
                else:
                    ids = ", ".join(["wd:" + x for x in property_id])
                    sparq_instanceof = f" ?item wdt:P31{class_tree} ?tree. \n FILTER (?tree in ({ids}))"

        else:
            sparq_instanceof = ""

        if "property_filters" in kwargs:
            for prop, value in kwargs["property_filters"].items():
                if value[0].lower() != "q":
                    raise ValueError(
                        f"Property value {value} is not a valid Wikidata ID"
                    )
                # TODO: wrap in optional
                sparq_property_filter += f"\n ?item wdt:{prop} wd:{value} ."

        endpoint_url = config.WIKIDATA_SPARQL_ENDPOINT
        query = f"""
        SELECT DISTINCT ?item ?itemLabel 
        WHERE
        {{
            {sparq_instanceof}
            {sparq_property_filter} 
            SERVICE wikibase:mwapi {{
                bd:serviceParam wikibase:api "EntitySearch" .
                bd:serviceParam wikibase:endpoint "www.wikidata.org" .
                bd:serviceParam mwapi:search "{text}" .
                bd:serviceParam mwapi:language "en" .
                ?item wikibase:apiOutputItem mwapi:item .
                ?num wikibase:apiOrdinal true .
              }}

            SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" .
            }}
        }} LIMIT {limit}
        """

        res = get_sparql_results(endpoint_url, query)["results"]["bindings"]

        res_df = pd.json_normalize(res)

        if len(res_df) > 0:
            res_df = res_df[["item.value", "itemLabel.value"]].rename(
                columns=lambda x: x.replace(".value", "")
            )

            res_df = res_df.reset_index().rename(columns={"index": "rank"})
            res_df["rank"] = res_df["rank"] + 1

            res_df = self.add_score_to_search_results_df(res_df, rank_col="rank")

        return res_df


class wikipedia_text_search(TextSearch):
    def __init__(self):
        super().__init__()

    def calculate_label_similarity(self, string1: str, string2: str) -> int:
        """
        Performs two checks: Levenshtein similarity, and that there is at least one token in common between the two
        strings. If there are no tokens in common the similarity is set to zero.

        Returns:
            int: similarity between 0 and 100. 
        """

        def tokenize(text) -> set:
            """Ignore text in brackets, and generate set of lowercase tokens"""
            no_brackets = re.sub(r"\([^)]*\)", "", text.lower())
            return set(re.findall(r"\w+(?:'\w+)?|[^\w\s,]", no_brackets))

        common_tokens = tokenize(string1).intersection(tokenize(string2))

        if len(common_tokens) == 0:
            return 0
        else:
            return fuzz.token_set_ratio(string1, string2)

    def run_search(self, text: str, limit=100, similarity_thresh=50, **kwargs):
        """
        Run Wikipedia search, then rank and limit results based on string similarity. 

        Args:
            text (str): text to search
            limit (int, optional): Defaults to 100.
            similarity_thresh (int, optional): The cut off to exclude items from search results. Defaults to 50. 

        Kwargs:
            instanceof_filter (str/list): property or properties to filter values by instance of. 
            include_class_tree (bool): whether to look in the subclass tree for the instance of filter.
            property_filters (dict): filters on exact values of properties you want to pass through. {property: value, ...}

        Returns:
            pd.DataFrame: columns rank, item, itemLabel, score
        """

        class_tree = "/wdt:P279*" if "include_class_tree" in kwargs else ""
        sparq_property_filter = ""

        if "instanceof_filter" in kwargs:
            property_id = kwargs["instanceof_filter"]
            if isinstance(property_id, str):
                # one instanceof in filter
                sparq_instanceof = f"?item wdt:P31{class_tree} wd:{property_id}."
            elif isinstance(property_id, list):
                if len(property_id) == 1:
                    sparq_instanceof = f"?item wdt:P31{class_tree} wd:{property_id[0]}."
                else:
                    ids = ", ".join(["wd:" + x for x in property_id])
                    sparq_instanceof = f" ?item wdt:P31{class_tree} ?tree. \n FILTER (?tree in ({ids}))"

        else:
            sparq_instanceof = ""

        if "property_filters" in kwargs:
            for prop, value in kwargs["property_filters"].items():
                if value[0].lower() != "q":
                    raise ValueError(
                        f"Property value {value} is not a valid Wikidata ID"
                    )
                sparq_property_filter += f"\n ?item wdt:{prop} wd:{value} ."

        endpoint_url = config.WIKIDATA_SPARQL_ENDPOINT
        query = f"""
        SELECT ?item ?wikipedia_title {{
            SERVICE wikibase:mwapi {{
                bd:serviceParam wikibase:endpoint "en.wikipedia.org" .
                bd:serviceParam wikibase:api "Generator" .
                bd:serviceParam mwapi:generator "search" .
                bd:serviceParam mwapi:gsrsearch "{text}" .
                bd:serviceParam mwapi:gsrlimit "max" .
                ?item wikibase:apiOutputItem mwapi:item . 
                ?wikipedia_title wikibase:apiOutput mwapi:title .
            }}
            hint:Prior hint:runFirst "true".
            {sparq_instanceof}
            {sparq_property_filter}
        }} LIMIT {limit}
        """

        res = get_sparql_results(endpoint_url, query)["results"]["bindings"]

        res_df = pd.json_normalize(res)

        if len(res_df) > 0:
            res_df = res_df[["item.value", "wikipedia_title.value"]].rename(
                columns=lambda x: x.replace(".value", "")
            )

            res_df["text_similarity"] = res_df["wikipedia_title"].apply(
                lambda s: self.calculate_label_similarity(text, s)
            )
            res_df = (
                res_df[res_df["text_similarity"] >= similarity_thresh]
                .sort_values("text_similarity", ascending=False)
                .reset_index(drop=True)
            )
            res_df = res_df.drop(columns="text_similarity")
            res_df = res_df.reset_index().rename(columns={"index": "rank"})
            res_df["rank"] = res_df["rank"] + 1

            res_df = self.add_score_to_search_results_df(res_df, rank_col="rank")

        return res_df


def combine_results(search_results: list, topn=20) -> pd.Series:
    """
    Combines list of dataframes returned from TextSearch objects, by taking a mean of the scores.

    Args:
        search_results (list)
        topn (int): the number of top results to return 

    Return:
        pd.Series: Wikidata references, ranked by score
    """

    all_results = pd.concat(search_results)

    if len(all_results) == 0:
        return pd.Series()

    score_series = (
        all_results.groupby("item").sum()["score"] / len(search_results)
    ).sort_values(ascending=False)

    if topn < len(score_series):
        score_series = score_series[0:topn]

    return score_series


def run(
    text: str,
    topn: int,
    limit=100,
    similarity_thresh=50,
    search_objects=[wikidata_text_search, wikipedia_text_search],
    **kwargs,
) -> pd.Series:
    """
    Run search for all the specified TextSearch objects and combine results.

    Args:
        text (str): text to search
        limit (int, optional): Defaults to 100.
        similarity_thresh (int, optional): The cut off to exclude items from search results. Defaults to 50. 

    Kwargs:
        instanceof_filter (str): the property to filter values by instance of. 
        include_class_tree (bool): whether to look in the subclass tree for the instance of filter.
        property_filters (dict): filters on exact values of properties you want to pass through. {property: value, ...}

    """

    # run all searches
    search_results = []

    for search in search_objects:
        s = search()
        res_df = s.run_search(
            text, limit, similarity_thresh=similarity_thresh, **kwargs
        )
        search_results.append(res_df)

    # concatenate results
    concat_results = combine_results(search_results, topn)

    return concat_results
