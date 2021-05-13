import re
import pandas as pd
from collections import Counter
from tqdm import tqdm
from ast import literal_eval
from typing import Union, List
import elasticsearch
from heritageconnector.disambiguation.search import es_text_search
from heritageconnector.config import config
from heritageconnector.utils.generic import paginate_list
from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.utils.wikidata import url_to_qid, filter_qids_in_class_tree
from heritageconnector.datastore import es
from heritageconnector import logging

logger = logging.get_logger(__name__)

tqdm.pandas()


class Reconciler:
    """
    For reconciling a categorical column in a DataFrame to a set of Wikidata QIDs. Uses a Wikidata Elasticsearch
    dump as well as SPARQL queries.

    The workflow using an instance of this class (here `rec`) is as follows:
    1. Run `rec.process_column` to perform controlled reconciliation of a DataFrame column against Wikidata. This will
    produce a DataFrame which maps the unique values in your column to Wikidata QIDs.
    2. Check that the values are mapped correctly. If you want to modify them, you can export them using `export_map_df_to_csv`
    and re-import them using `import_map_df_from_csv`.
    3. Run `create_column_from_map_df` on the original column to produce a new column (pd.Series) with values mapped to Wikidata QIDs.
    """

    def __init__(
        self,
        es_connector: Union[elasticsearch.client.Elasticsearch, str] = "from_config",
        es_index: str = "from_config",
    ):
        self.es_connector = es if es_connector == "from_config" else es_connector
        self.es_index = (
            config.ELASTIC_SEARCH_WIKI_INDEX if es_index == "from_config" else es_index
        )
        self.search = es_text_search(self.es_index, self.es_connector)

    @staticmethod
    def _get_subject_items_from_pid(pid: str) -> list:
        """
        Gets a list of subject items from a Wikidata property ID using 'subject
        item of this property (P1629)'. If a URL is passed extracts the PID from
        it if it exists, else raises a ValueError.
        """

        if pid.startswith("http"):
            logger.warning("WARNING: URL instead of PID entered. Converting to PID")
            pids = re.findall(r"(P\d+)", pid)

            if len(pids) == 1:
                pid = pids[0]
            else:
                raise ValueError("URL not a valid property URL.")

        query = f"""
        SELECT ?property WHERE {{
        wd:{pid} wdt:P1629 ?property.
        }}
        """

        res = get_sparql_results(config.WIKIDATA_SPARQL_ENDPOINT, query)

        if "results" in res:
            bindings = res["results"]["bindings"]
            qids = [url_to_qid(item["property"]["value"]) for item in bindings]

            return qids
        else:
            return []

    def _lookup_value(
        self,
        text: str,
        search_limit_per_item: int = 5000,
        text_similarity_thresh: int = 95,
        field_exists_filter: str = None,
    ) -> List[str]:
        """Lookup text and return list of QIDs"""
        qids = self.search.run_search(
            # limit is large here as we want to fetch all the results then filter them by
            # text similarity later
            text,
            limit=search_limit_per_item,
            return_instanceof=False,
            similarity_thresh=text_similarity_thresh,
            field_exists_filter=field_exists_filter,
            return_exact_only=True,
        )

        return qids

    def process_column(
        self,
        column: pd.Series,
        multiple_vals: bool,
        pid: str = None,
        class_include: Union[str, list] = None,
        class_exclude: Union[str, list] = None,
        search_args: dict = {
            "search_limit_per_item": 5000,
            "text_similarity_thresh": 95,
            "field_exists_filter": None,
        },
    ) -> pd.Series:
        """
        Run reconciliation on a categorical column.

        Args:
            column (pd.Series): column to reconcile.
            multiple_vals (bool): whether the column contains multiple values per record
                If values are: lists -> True, strings -> False.
            pid (str, Optional): Wikidata PID of the selected column. Only needed to look up a class
                constraint using 'subject item of this property' (P1629).
            class_include (Union[str, list], Optional): class tree to look under. Will be ignored if
                PID is specified. Defaults to None.
            class_exclude (Union[str, list], Optional): class trees containing this class (above the entity)
                will be excluded. Defaults to None.
            search_args (dict, Optional): keyword arguments for text search. See `heritageconnector.disambiguation.search.es_text_search`
                for the full set of arguments.
        """

        # if PID is specified, get the classes to include using the 'subject item of this property' (P1629)
        if pid and (not class_include):
            class_include = self._get_subject_items_from_pid(pid)

        if multiple_vals:
            all_vals = column.sum()
        else:
            all_vals = column.astype(str).tolist()

        all_vals = [i for i in all_vals if i != ""]

        # using Counter here allows us to cut off low frequency values in future
        val_count = pd.Series(Counter(all_vals))
        map_df = pd.DataFrame(val_count).rename(columns={0: "count"})

        logger.info("Looking up Wikidata QIDs against Elasticsearch Wikidata dump")

        map_df["qids"] = map_df.index.to_series().progress_apply(
            lambda v: self._lookup_value(v, **search_args)
        )

        # get set of types to look up in subclass tree
        instanceof_unique = set(map_df["qids"].sum())

        # return only values that exist in subclass tree
        logger.info(f"Filtering to values in subclass tree of {class_include}")
        # 50 seems like a sensible size given this is the page size commonly used on the wb APIs
        instanceof_unique_paginated = paginate_list(
            list(instanceof_unique), page_size=50
        )
        instanceof_filtered = []

        for page in tqdm(instanceof_unique_paginated):
            instanceof_filtered += filter_qids_in_class_tree(
                page, class_include, class_exclude
            )

        # filter found QIDs by those in instanceof_filtered
        map_df["filtered_qids"] = map_df["qids"].apply(
            lambda l: [i for i in l if i in instanceof_filtered]
        )

        return map_df


def export_map_df_to_csv(map_df: pd.DataFrame, file_path: str = None):
    """
    Export dataframe of unique column values and their reconciled QIDs to a tab-separated CSV file.

    Args:
        map_df (pd.DataFrame): mapping DataFrame created using `Reconciler().process_column`
        file_path (str): path to export CSV to
    """

    map_df.sort_values("count", ascending=False).to_csv(file_path, sep="\t")

    logger.info(f"Dataframe of value to entity mappings exported to {file_path}")


def import_map_df_from_csv(file_path) -> pd.DataFrame:
    """
    Import previously exported CSV of unique column values and their reconciled QIDs.

    Args:
        file_path (str): path to import CSV from
    """

    map_df = pd.read_csv(file_path, sep="\t", index_col=0)

    # turn list columns back into lists (from strings)
    for col in ["qids", "filtered_qids"]:
        map_df[col] = map_df[col].apply(literal_eval)

    return map_df


def create_column_from_map_df(
    original_column: pd.Series, map_df: pd.DataFrame, multiple_vals: bool
) -> pd.Series:
    """
    Uses a mapping DataFrame created by `Reconciler().process_column` to map the values in a Series (`original_column`) to a new column,
    which contains QIDs.

    Args:
        original_column (pd.Series): column to apply the transform to
        map_df (pd.DataFrame): DataFrame containing mapping between values in the column and QIDs, created
        by `Reconciler().process_column`
        multiple_vals (bool): whether there are multiple values in each cell of `original_column`. This value should be the same
        as the one provided to `Reconciler().process_column`.

    Returns:
        pd.Series: transformed column
    """

    if multiple_vals:
        new_col = original_column.progress_apply(
            lambda x: map_df.loc[
                [i for i in x if len(i) > 0], "filtered_qids"
            ].values.sum()
            if x != [""]
            else []
        )

        return new_col.apply(lambda x: [] if str(x) == "[][]" else x)
    else:
        return original_column.progress_apply(
            lambda x: map_df.loc[x, "filtered_qids"] if x != "" else []
        )
