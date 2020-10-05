import re
import pandas as pd
from collections import Counter
from tqdm import tqdm
from typing import Union
from heritageconnector.disambiguation.search import es_text_search
from heritageconnector.config import config, field_mapping
from heritageconnector.utils.generic import paginate_list
from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.utils.wikidata import url_to_qid, filter_qids_in_class_tree
from heritageconnector import logging

logger = logging.get_logger(__name__)

tqdm.pandas()


class reconciler:
    def __init__(self, dataframe: pd.DataFrame, table: str):
        self.df = dataframe
        self.table = table.upper()

    def get_column_pid(self, column: str) -> str:
        """
        Retrieves the column PID from field mapping config. Returns None if one is not present.
        """

        config_table = field_mapping.mapping[self.table]

        if column not in config_table:
            raise KeyError(
                f"column {column} not in field mapping for table {self.table}"
            )
        else:
            config_column = config_table[column]

            if "PID" not in config_column:
                logger.warn(
                    f"WARNING: PID has not been specified for column {column} in table {self.table}"
                )

                return None
            else:
                return config_column["PID"]

    @staticmethod
    def get_subject_items_from_pid(pid: str) -> list:
        """
        Gets a list of subject items from a Wikidata property ID using 'subject
        item of this property (P1629)'. If a URL is passed extracts the PID from 
        it if it exists, else raises a ValueError.
        """

        if pid.startswith("http"):
            logger.warn("WARNING: URL instead of PID entered. Converting to PID")
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

    def process_column(
        self,
        column: str,
        multiple_vals: bool,
        pid: str = None,
        class_include: Union[str, list] = None,
        class_exclude: Union[str, list] = None,
        search_limit_per_item: int = 5000,
        text_similarity_thresh: int = 95,
        field_exists_filter: str = None,
    ) -> pd.Series:
        """
        Run reconciliation on a categorical column.

        Args:
            column (str): column to reconcile.
            multiple_vals (bool): whether the column contains multiple values per record. 
                If values are: lists -> True, strings -> False.
            pid (str, Optional): Wikidata PID of the selected column. Only needed to look up a class 
                constraint using 'subject item of this property' (P1629).
            class_include (Union[str, list], Optional): class tree to look under. Will be ignored if 
                PID is specified. Defaults to None.
            class_exclude (Union[str, list], Optional): class trees containing this class (above the entity)
                will be excluded. Defaults to None.
            search_limit_per_item (int): Number of results to return from the Elasticsearch index per search.
                Set lower (~200) for speed if queries are more unique. With a small limit some results for generic
                queries may be missed. Defaults to 5000.
            text_similarity_thresh (int). Text similarity threshold for a match. Defaults to 95.
            field_exists_filter (str, optional): if specified, all searches will be filtered to only documents which have a value for this field.
                For example to filter to documents which have a P279 value, set `field_exists_filter = "claims.P279"`. Defaults to None.

        """

        if column not in self.df.columns:
            raise ValueError("Column not in dataframe columns.")

        # get PID and lookup filter (entity type) for PID
        if not class_include and not pid:
            pid = self.get_column_pid(column)
            class_include = self.get_subject_items_from_pid(pid)
        elif not class_include:
            class_include = self.get_subject_items_from_pid(pid)

        if multiple_vals:
            all_vals = self.df[column].sum()
        else:
            all_vals = self.df[column].astype(str).tolist()

        all_vals = [i for i in all_vals if i != ""]

        # using Counter here allows us to cut off low frequency values in future
        val_count = pd.Series(Counter(all_vals))
        map_df = pd.DataFrame(val_count).rename(columns={0: "count"})

        #  look up QIDs for unique values
        search = es_text_search()

        def lookup_value(text):
            qids = search.run_search(
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

        logger.info(
            "Looking up Wikidata qcodes for items on Elasticsearch Wikidata dump"
        )

        map_df["qids"] = map_df.index.to_series().progress_apply(lookup_value)
        self._map_df = map_df

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

        self.instanceof_filtered = instanceof_filtered

        # filter found QIDs by those in instanceof_filtered
        map_df["filtered_qids"] = map_df["qids"].apply(
            lambda l: [i for i in l if i in instanceof_filtered]
        )

        self._map_df = map_df

        return self.create_column_from_map_df(column, map_df, multiple_vals)

    def create_column_from_map_df(
        self, original_column: str, map_df: pd.DataFrame, multiple_vals: bool
    ) -> pd.Series:
        """
        Creates a column 

        Args:
            original_column (str): column to apply the transform to
            map_df (pd.DataFrame): dataframe containing unique column values and their mapping to Wikidata entities
            multiple_vals (bool): whether items in the original dataframe column contain multiple values per element
                (elements are lists)

        Returns:
            pd.Series: transformed column
        """

        if multiple_vals:
            return self.df[original_column].apply(
                lambda x: map_df.loc[
                    [i for i in x if i != ""], "filtered_qids"
                ].values.sum()
                if x != [""]
                else []
            )
        else:
            return self.df[original_column].apply(
                lambda x: map_df.loc[x, "filtered_qids"] if x != "" else []
            )
