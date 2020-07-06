from heritageconnector.nlp.string_pairs import fuzzy_match
from fuzzywuzzy import fuzz
from heritageconnector.base.disambiguation import TextSearch
from heritageconnector.config import config
from heritageconnector.utils.sparql import get_sparql_results
import pandas as pd


class wikidata_text_search(TextSearch):
    def __init__(self):
        super().__init__()

    def run_search(self, text: str, limit=100, **kwargs):
        """
        [summary]

        Args:
            text (str): [description]
            limit (int, optional): [description]. Defaults to 100.

        Returns:
            [type]: [description]
        """

        class_tree = "/wdt:P279*" if "include_class_tree" in kwargs else ""
        sparq_property_filter = ""

        if "instanceof_filter" in kwargs:
            property_id = kwargs["instanceof_filter"]
            sparq_instanceof = f"?item wdt:P31{class_tree} wd:{property_id}."

        if "property_filters" in kwargs:
            for prop, value in kwargs["property_filters"].items():
                if value[0].lower() != "q":
                    raise ValueError(
                        f"Property value {value} is not a valid Wikidata ID"
                    )
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

        res_df = pd.json_normalize(res)[["item.value", "itemLabel.value"]].rename(
            columns=lambda x: x.replace(".value", "")
        )
        res_df = res_df.reset_index().rename(columns={"index": "rank"})
        res_df["rank"] = res_df["rank"] + 1

        max_rank = len(res_df)
        if max_rank == 1:
            res_df.at[0, "score"] = 1
        else:
            res_df["score"] = res_df["rank"].apply(lambda x: max_rank - x)
            sum_confidence = res_df["score"].sum()
            res_df["score"] = res_df["score"].apply(lambda x: x / sum_confidence)

        return res_df
