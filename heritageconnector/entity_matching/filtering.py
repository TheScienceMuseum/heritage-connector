import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz
from itertools import compress
import re
from heritageconnector.nlp.string_pairs import fuzzy_match
from heritageconnector.utils.wikidata import entities
from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.utils.generic import add_dicts
from heritageconnector.utils.data_transformation import get_year_from_date_value
from heritageconnector.config import config
from heritageconnector import logging

logger = logging.get_logger(__name__)


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
        self.date_values = ["birthYear", "deathYear", "inceptionYear", "dissolvedYear"]

        self.df.loc[:, self.qcode_col] = self._clean_qcode_col()
        self.qcodes_unique = list(set(self.df[self.qcode_col].sum()))
        self.sparql_endpoint_url = config.WIKIDATA_SPARQL_ENDPOINT

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
        SELECT ?item ?itemLabel ?altLabel ?birthYear ?deathYear ?inceptionYear ?dissolvedYear
                WHERE
                {{
                    VALUES (?item) {{ {map_ids(qcodes)} }}
                    {sparq_instanceof}
                    OPTIONAL{{
                        ?item wdt:P569 ?birthDate.
                        BIND( year(?birthDate) AS ?birthYear )
                        }}
                    OPTIONAL {{
                        ?item wdt:P570 ?deathDate.
                        BIND( year(?deathDate) AS ?deathYear )
                        }}
                    OPTIONAL {{
                        ?item wdt:P571 ?inceptionDate.
                        BIND( year(?inceptionDate) AS ?inceptionYear )
                        }}
                    OPTIONAL {{
                        ?item wdt:P576 ?dissolvedDate.
                        BIND( year(?dissolvedDate) AS ?dissolvedYear )  
                        }}
                    OPTIONAL {{
                        ?item skos:altLabel ?altLabel .
                        FILTER (lang(?altLabel) = "en")
                        }}

                    SERVICE wikibase:label {{ 
                    bd:serviceParam wikibase:language "en" .
                    }}
                }}
        """
        self.query = query
        res = get_sparql_results(self.sparql_endpoint_url, query)["results"]["bindings"]

        res_df = pd.json_normalize(res)
        res_df.loc[:, "qcode"] = res_df["item.value"].apply(
            lambda x: re.findall(r"(Q\d+)", x)[0]
        )

        # fill missing columns with blanks for any columns that aren't in the data
        final_cols = [
            "qcode",
            "itemLabel.value",
            "altLabel.value",
            "birthYear.value",
            "deathYear.value",
            "inceptionYear.value",
            "dissolvedYear.value",
        ]
        cols_missing = set(final_cols) - set(res_df.columns.values.tolist())
        for col in cols_missing:
            res_df[col] = ""

        res_df = res_df[
            [
                "qcode",
                "itemLabel.value",
                "altLabel.value",
                "birthYear.value",
                "deathYear.value",
                "inceptionYear.value",
                "dissolvedYear.value",
            ]
        ]

        # convert aliases to lowercase and fill nan with empty string
        res_df["altLabel.value"] = (
            res_df["altLabel.value"].fillna("").astype(str).str.lower()
        )

        res_df = res_df.drop_duplicates()

        # rename columns (remove .value suffic from year columns)
        res_df = res_df.rename(
            columns={"itemLabel.value": "label", "altLabel.value": "alias"}
        )
        res_df = res_df.rename(columns=lambda x: x.replace(".value", ""))
        self.sparql_res = res_df
        return res_df

    def add_label_filter(
        self,
        col,
        include_aliases=False,
        threshold=90,
        fuzzy_match_scorer=fuzz.token_sort_ratio,
    ):
        """
        Add a fuzzy matching filter on a column name and the label of each page.

        Args:
            col ([type]): [description]
        """

        new_filter = {
            "label": {
                "label_col": col,
                "include_aliases": include_aliases,
                "threshold": threshold,
                "fuzzy_match_scorer": fuzzy_match_scorer,
            }
        }
        logger.info(f"Added filter {new_filter}")
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
        logger.info(f"Added filter {new_filter}")
        self.filters.update(new_filter)

    def add_date_filter(self, date_col: str, wiki_value: str, uncertainty: int):
        """
        Adds a filter on records by year in a date column. Uncertainty allows dates from Wikidata +-
        a certain number of years outside the recorded date.

        Args:
            date_col (str): 
            wiki_value (str): from birthDate, deathDate, inceptionDate, dissolvedDate
            uncertainty (int): number of years either side of the recorded date to accept a date from Wikidata
        """

        assert wiki_value in self.date_values

        new_filter = {
            f"date_{wiki_value}": {
                "date_col": date_col,
                "wiki_value": wiki_value,
                "uncertainty": uncertainty,
            }
        }

        logger.info(f"Added filter {new_filter}")
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

    def _get_labels(self, res_df: pd.DataFrame, qcodes: list) -> list:
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

    def _get_dates(self, wiki_value: str, res_df: pd.DataFrame, qcodes: list) -> dict:
        """
        Get dates for qcodes from specified date_col in res_df. If the specified date value doesn't exist
        for that qcode, then an empty list is returned for the key. 

        Args:
            wiki_value (str)
            res_df (pd.DataFrame)
            qcodes (list)

        Raises:
            Exception

        Returns:
            dict: keys=qcodes, vals=date or None
        """

        date_dict = dict()

        for qcode in qcodes:
            dates_unique = list(
                set(res_df.loc[res_df["qcode"] == qcode, wiki_value].dropna().tolist())
            )

            if len(dates_unique) == 1:
                date_dict.update({qcode: dates_unique[0]})
            elif len(dates_unique) == 0:
                date_dict.update({qcode: None})
            elif len(dates_unique) > 1:
                #  more than one birth date in Wikidata: return an average
                dates_unique = [int(date) for date in dates_unique]
                date_average = int(sum(dates_unique) / len(dates_unique))
                date_dict.update({qcode: str(date_average)})

        return date_dict

    def _apply_date_filter(
        self, res_df: pd.DataFrame, row: pd.Series, qcode_col: str, filter_args: dict
    ) -> list:
        """
        Apply date filter which filters records by year of the field specified, with a specified amount
        of leeway expressed through an uncertainty value.

        Args:
            res_df (pd.DataFrame): 
            row (pd.Series): 
            filter_args (dict): col, wiki_value, uncertainty

        Returns:
            list: filtered qcodes
        """

        date_col, wiki_value, uncertainty = (
            filter_args["date_col"],
            filter_args["wiki_value"],
            filter_args["uncertainty"],
        )
        record_year = get_year_from_date_value(row[date_col])

        qcodes = row[qcode_col]
        qcode_years = self._get_dates(wiki_value, res_df, qcodes)

        # if a year can't be extracted from the column in question return all the qcodes
        if not record_year:
            return qcodes

        # return all years within the specified uncertainty of the target year, or those for which there wasn't a date
        # on Wikidata (value in qcode_years is None)
        return [
            k
            for k, v in qcode_years.items()
            if v is None or -uncertainty <= int(record_year) - int(v) <= uncertainty
        ]

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

        label_col, threshold, include_aliases, fuzzy_match_scorer = (
            filter_args["label_col"],
            filter_args["threshold"],
            filter_args["include_aliases"],
            filter_args["fuzzy_match_scorer"],
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
                fuzzy_match(
                    source_string,
                    target,
                    scorer=fuzzy_match_scorer,
                    threshold=threshold,
                )
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

        logger.info("Running Wikidata query..")
        sparql_res = self._run_wikidata_query(
            self.qcodes_unique,
            instanceof_filter,
            property_id=property_id,
            include_class_tree=include_class_tree,
        )
        self.sparql_res = sparql_res

        logger.debug("Applying filters...")
        self.df.loc[:, self.new_qcode_col] = self.df[self.qcode_col]

        # we only need to process the rows which don't already have an empty list
        df_to_process = self.df[self.df[self.new_qcode_col].map(lambda d: len(d)) > 0]

        if instanceof_filter:
            logger.info(f"Filter: instance of {property_id}")
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
            logger.info(
                f"Filter: check label similarity against column {label_filter_args['label_col']}"
            )

            for idx, row in tqdm(
                df_to_process.iterrows(), total=df_to_process.shape[0]
            ):
                self.df.at[idx, self.new_qcode_col] = self._apply_label_filter(
                    sparql_res, row, self.new_qcode_col, label_filter_args
                )

        if any("date" in key for key in self.filters.keys()):
            date_filters = [k for k in self.filters.keys() if "date" in k]

            for f in date_filters:
                df_to_process = self.df[
                    self.df[self.new_qcode_col].map(lambda d: len(d)) > 0
                ]
                date_filter_args = self.filters[f]
                logger.info(f"Filter: date ({date_filter_args['date_col']})")
                for idx, row in tqdm(
                    df_to_process.iterrows(), total=df_to_process.shape[0]
                ):
                    self.df.at[idx, self.new_qcode_col] = self._apply_date_filter(
                        sparql_res, row, self.new_qcode_col, date_filter_args
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
            logger.warn(
                "WARNING: filters not run against dataframe yet so nothing returned."
            )

    def view_stats(self):
        """
        Prints statistics for the filtering process.
        """

        if self.new_qcode_col not in self.df.columns:
            raise Exception("Filters not run so no stats to display.")

        num_records_after_filter = len(
            self.df[(self.df[self.new_qcode_col].map(lambda d: len(d)) > 0)]
        )
        num_records_total = len(self.df)
        perc_records_with_unique_match = round(
            100 * num_records_after_filter / num_records_total, 1
        )

        logger.info(
            f"No. records after filtering: {num_records_after_filter}/{num_records_total} ({perc_records_with_unique_match}%)"
        )

    def view_filters(self):
        """
        Shows filters added so far. For use in an interactive environment.
        """

        logger.info("Filters: ")

        for k, v in self.filters.items():
            logger.info(f" - {k}: {v}")

    # TODO: instance filters from_file, to_file
