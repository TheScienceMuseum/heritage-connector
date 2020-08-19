from heritageconnector.disambiguation.search import wikidata_text_search
from heritageconnector.config import config, field_mapping
from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.utils.wikidata import url_to_qid

import re
import pandas as pd
from collections import Counter
from tqdm import tqdm

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
                print(
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
            # print("WARNING: URL instead of PID entered. Converting to PID")
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

    def process_column(self, column: str, multiple_vals: bool) -> pd.Series:
        """
        Run reconciliation on a categorical column.

        Args:
            column (str)
            multiple_vals (bool): whether the column contains multiple values per record. 
            If values are: lists -> True, strings -> False.
        """

        if column not in self.df.columns:
            raise ValueError("Column not in dataframe columns.")

        # get PID and lookup filter (entity type) for PID
        pid = self.get_column_pid(column)
        lookup_filter = self.get_subject_items_from_pid(pid)

        if multiple_vals:
            all_vals = self.df[column].sum()
        else:
            all_vals = self.df[column].astype(str).tolist()

        all_vals = [i for i in all_vals if i != ""]

        # using Counter here allows us to cut off low frequency values in future
        val_count = pd.Series(Counter(all_vals))
        map_df = pd.DataFrame(val_count).rename(columns={0: "count"})

        #  look up QIDs for unique values
        search = wikidata_text_search()

        def lookup_value(text):
            res_df = search.run_search(
                text, instanceof_filter=lookup_filter, include_class_tree=True
            )
            if len(res_df) == 0:
                return []
            else:
                return [url_to_qid(i) for i in res_df["item"].tolist()]

        print("Looking up Wikidata qcodes for unique items..")
        map_df["qid"] = map_df.index.to_series().progress_apply(lookup_value)

        if multiple_vals:
            return self.df[column].apply(
                lambda x: map_df.loc[[i for i in x if i != ""], "qid"].values.sum()
                if x != [""]
                else []
            )
        else:
            return self.df[column].apply(
                lambda x: map_df.loc[x, "qid"].values if x != "" else []
            )
