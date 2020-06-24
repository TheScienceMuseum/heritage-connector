from heritageconnector.nlp.string_pairs import fuzzy_match
from heritageconnector.utils.wikidata import entities
from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.utils.generic import add_dicts
import pandas as pd
from tqdm import tqdm
from itertools import compress
import re


class Filter:
    def __init__(self, dataframe: pd.DataFrame, qcode_col: str):
        """
        Set of filters to filter candidate qcode matches from a dataframe. 

        Args:
            dataframe (pd.DataFrame): each row is a record, contains candidate qcode matches
            qcode_col (str): name of column containing qcode candidate matches in format list(str, str, ...)
        """

        self.df = dataframe
        self.filters = {}
        self.qcode_col = qcode_col
        self.new_qcode_col = "qcodes_filtered"

        self.df.loc[:, self.qcode_col] = self._clean_qcode_col()
        self.qcodes_unique = list(set(self.df[self.qcode_col].sum()))
        self.sparql_endpoint_url = "https://query.wikidata.org/sparql"

    def _clean_qcode_col(self):
        """
        ensure each qcode in self.qcode_col starts with a Q, and reduce each cell to only 
        contain unique qcodes
        """

        new_col = self.df[self.qcode_col].apply(
            lambda item: [i for i in item if str(i).startswith("Q")]
        )
        new_col = new_col.apply(lambda item: list(set(item)))

        return new_col

    def _load_entities_instance(self):
        """
        Loads a heritageconnector.utils.wikidata.entities instance, used for finding properties and values from Wikidata. 
        """

        return entities(self.qcodes_unique, lang=self.lang)

    def _run_wikidata_query_paginated(
        self, page_limit, qcodes: list, instance_of_filter: bool, **kwargs
    ):
        qcodes_paginated = [
            qcodes[i : i + page_limit] for i in range(0, len(qcodes), page_limit)
        ]

        return pd.concat(
            [
                self._run_wikidata_query(qcodes, instance_of_filter, **kwargs)
                for page in qcodes_paginated
            ]
        )

    def _run_wikidata_query(
        self, qcodes: list, instanceof_filter: bool, **kwargs
    ) -> pd.DataFrame:
        """
        Runs a parametrised Wikidata query with options to filter by instance or subclass of a specific property.
        Returns a dataframe with qcodes matching the filter, their labels and their aliases. 

        Args:
            qcodes (list): a list of qcodes before filtering
            instanceof_filter (bool): whether to filter results by instance or subclass of a certain property

        Kwargs:
            property_id (str): the property to use as a parameter for the 'instance of' filter
            include_class_tree (bool): whether to include all subclasses in the search up the tree, or just the instanceof property

        Returns:
            pd.DataFrame: columns are qcode, label, alias
        """

        if instanceof_filter:
            # process kwargs related to the 'instance of' filter
            try:
                property_id = kwargs["property_id"]
            except KeyError:
                raise ValueError(
                    "Keyword argument property_id (str) must be passed if using instance_of_filter."
                )

            try:
                include_class_tree = kwargs["include_class_tree"]
            except KeyError:
                raise ValueError(
                    "Keyword argument include_class_tree (bool) must be passed if using instance_of_filter."
                )

            # create line of SPARQL query that does filtering
            class_tree = "/wdt:P279*" if include_class_tree else ""
            sparq_instanceof = f"?item wdt:P31{class_tree} wd:{property_id}."

        else:
            sparq_instanceof = ""

        def map_ids(ids):
            return " ".join([f"(wd:{i})" for i in ids])

        query = f"""
        SELECT ?item ?itemLabel ?altLabel
                WHERE
                {{
                    VALUES (?item) {{ {map_ids(qcodes)} }}
                    {sparq_instanceof}
                    OPTIONAL {{
                        ?item skos:altLabel ?altLabel .
                        FILTER (lang(?altLabel) = "en")
                        }}

                    SERVICE wikibase:label {{ 
                    bd:serviceParam wikibase:language "en" .
                    }}
                }} 
        GROUP BY ?item ?itemLabel ?altLabel
        """
        self.query = query
        res = get_sparql_results(self.sparql_endpoint_url, query)["results"]["bindings"]

        res_df = pd.json_normalize(res)
        res_df.loc[:, "qcode"] = res_df["item.value"].apply(
            lambda x: re.findall(r"(Q\d+)", x)[0]
        )
        res_df = res_df[["qcode", "itemLabel.value", "altLabel.value"]]

        # convert aliases to lowercase and drop duplicates
        res_df["altLabel.value"] = (
            res_df["altLabel.value"].fillna(" ").astype(str).str.lower()
        )
        res_df = res_df.drop_duplicates()

        res_df = res_df.rename(
            columns={"itemLabel.value": "label", "altLabel.value": "alias"}
        )

        return res_df

    def add_label_filter(self, col, threshold=90, include_aliases=False):
        """
        Add a fuzzy matching filter on a column name and the label of each page.

        Args:
            col ([type]): [description]
        """

        new_filter = {
            "label": {
                "label_col": col,
                "threshold": threshold,
                "include_aliases": include_aliases,
            }
        }
        print(f"Added filter {new_filter}")
        self.filters.update(new_filter)

    def add_instanceof_filter(self, property_id: str, include_class_tree: bool):
        """
        Add a filter to return only records of a given type, indicated by the 
        'instance of' property or its subclass tree. 

        Args:
            property_id (str): the property to filter by
            include_class_tree (bool): whether to include the class hierarchy as well 
            as the 'instance of' property
        """

        new_filter = {
            "instance_of": {
                "property_id": property_id,
                "include_class_tree": include_class_tree,
            }
        }
        print(f"Added filter {new_filter}")
        self.filters.update(new_filter)

    def _get_aliases(self, res_df: pd.DataFrame, qcodes: list) -> list:
        """
        Get all aliases for supplied qcodes.

        Args:
            res_df (dataframe): result of self.process_dataframe
            qcodes (list): list of qcodes

        Returns:
            dict
        """
        return {
            qcode: res_df.loc[res_df["qcode"] == qcode, "alias"].tolist()
            for qcode in qcodes
        }

    def _get_labels(self, res_df: pd.DataFrame, qcodes: list):
        """
        Get all labels for supplied qcodes.

        Args:
            res_df (dataframe): result of self.process_dataframe
            qcodes (list): list of qcodes

        Returns:
            dict
        """
        return {
            qcode: res_df.loc[res_df["qcode"] == qcode, "label"].unique().tolist()
            for qcode in qcodes
        }

    def _apply_label_filter(
        self, res_df: pd.DataFrame, row: pd.Series, qcode_col: str, filter_args: dict
    ) -> list:
        """
        Apply 

        Args:
            res_df (pd.DataFrame): result of sparql query with headers qcode, label, alias
            row (pd.Series): a row of self.df
            qcode_col (str): the column of the row containing qcodes to filter
            filter_args (dict): label_col, threshold, include_aliases

        Returns:
            list: filtered qcodes
        """

        label_col, threshold, include_aliases = (
            filter_args["label_col"],
            filter_args["threshold"],
            filter_args["include_aliases"],
        )

        source_string = row[label_col]
        qcodes = row[qcode_col]

        labels = self._get_labels(res_df, qcodes)

        if include_aliases:
            aliases = self._get_aliases(res_df, qcodes)
            # add aliases list onto labels list for each key
            labels = add_dicts(labels, aliases)

        qcodes_matched = []
        # for each qcode, produce a boolean list of matches between source and all targets.
        # add the qcode to qcodes_matched if at least on one of the matches returned True
        for qcode in qcodes:
            text_matches = [
                fuzzy_match(source_string, target, threshold=threshold)
                for target in labels[qcode]
            ]
            if any(x for x in text_matches):
                qcodes_matched.append(qcode)

        return qcodes_matched

    def _apply_instanceof_filter(
        self, qcodes_unique_filtered: list, row: pd.Series, qcode_col: str
    ) -> list:
        """
        Using a filtered list returned from the sparql query, 

        Args:
            qcodes_unique_filtered (list): unique list of qcodes returned from the sparql query (after filtering)
            row (pd.Series): a row of self.df
            qcode_col (str): the column of the row containing qcodes to filter

        Returns:
            list
        """

        return [i for i in row[qcode_col] if i in qcodes_unique_filtered]

    def process_dataframe(self) -> pd.DataFrame:
        """
        Processes dataframe with filters created and returns the processed dataframe.
        TODO: Also shows a message of how many have matched.

        Returns:
            pd.DataFrame: [description]
        """

        if "instance_of" in self.filters:
            instanceof_filter = True
            property_id = self.filters["instance_of"]["property_id"]
            include_class_tree = self.filters["instance_of"]["include_class_tree"]
        else:
            instanceof_filter = property_id = include_class_tree = False

        print("Running Wikidata query..")
        sparql_res = self._run_wikidata_query(
            self.qcodes_unique,
            instanceof_filter,
            property_id=property_id,
            include_class_tree=include_class_tree,
        )
        self.sparql_res = sparql_res

        print("Applying filters...")
        self.df.loc[:, self.new_qcode_col] = self.df[self.qcode_col]

        # we only need to process the rows which don't already have an empty list
        df_to_process = self.df[self.df[self.new_qcode_col].map(lambda d: len(d)) > 0]

        if instanceof_filter:
            print(f"Filter: instance of {property_id}")
            qcodes_unique_filtered = sparql_res["qcode"].unique().tolist()

            for idx, row in tqdm(
                df_to_process.iterrows(), total=df_to_process.shape[0]
            ):
                self.df.at[idx, self.new_qcode_col] = self._apply_instanceof_filter(
                    qcodes_unique_filtered, row, self.new_qcode_col
                )

        # we can do this again each time we run a new filter
        df_to_process = self.df[self.df[self.new_qcode_col].map(lambda d: len(d)) > 0]

        if "label" in self.filters:
            label_filter_args = self.filters["label"]
            print(
                f"Filter: check label similarity against column {label_filter_args['label_col']}"
            )

            for idx, row in tqdm(
                df_to_process.iterrows(), total=df_to_process.shape[0]
            ):
                self.df.at[idx, self.new_qcode_col] = self._apply_label_filter(
                    sparql_res, row, self.new_qcode_col, label_filter_args
                )

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the dataframe

        Returns:
            pd.DataFrame
        """

        if self.new_qcode_col in self.df.columns:
            return self.df
        else:
            print("WARNING: filters not run against dataframe yet so nothing returned.")

    def view_filters(self):
        """
        Shows filters added so far. For use in an interactive environment.
        """

        print("Filters: ")

        for k, v in self.filters.items():
            print(f" - {k}: {v}")

    # TODO: add_date_filter,
    # TODO: instance filters from_file, to_file
