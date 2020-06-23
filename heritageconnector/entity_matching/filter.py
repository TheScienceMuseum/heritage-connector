from heritageconnector.nlp.string_pairs import fuzzy_match
from heritageconnector.utils.wikidata import entities
import pandas as pd
from tqdm import tqdm
from itertools import compress


class Filter:
    def __init__(self, dataframe: pd.DataFrame, qcode_col: str, lang="en"):
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
        self.lang = lang

        self.qcodes_unique = self._get_unique_qcodes()
        self.entities = self._load_entities_instance()
        self.df[self.qcode_col] = self._clean_qcode_col()

    def _get_unique_qcodes(self):
        """
        get unique list of qcodes
        """

        qcodes_unique = list(set(self.df[self.qcode_col].sum()))
        qcodes_unique = [i for i in qcodes_unique if str(i).startswith("Q")]

        return qcodes_unique

    def _clean_qcode_col(self):
        """
        ensure each qcode in self.qcode_col starts with a Q, and reduce to unique values
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

    def add_property_filter(self, property_id, value, column=False):
        """
        Add a filter which asserts a match between a column or string, and a value 
        of a given property ID.

        Args:
            value ([type]): [description]
            property_id ([type]): [description]
            column (bool, optional): [description]
        """

        new_filter = {"property": [property_id, value, column]}
        print(f"Added filter {new_filter}")
        self.filters.update(new_filter)

    def add_label_filter(self, col, threshold=90, include_aliases=False):
        """
        Add a fuzzy matching filter on a column name and the label of each page.

        Args:
            col ([type]): [description]
        """

        new_filter = {"label": [col, threshold, include_aliases]}
        print(f"Added filter {new_filter}")
        self.filters.update(new_filter)

    def _apply_property_filter(
        self, qcode_col: str, filter_params: list, row: pd.Series
    ) -> list:
        """
        Apply a property filter to a row. Filters qcodes on whether a specified property has or contains
        a certain value.

        Args:
            qcode_col (str): the column name in the row to apply the filter to
            filter_params (list)
            row (pd.Series)

        Returns:
            list: list of qcodes after applying filter
        """

        property_id, value, column = filter_params
        qcodes = row[qcode_col]

        if column:
            match_val = row[value]
        else:
            match_val = value

        try:
            property_vals = self.entities.get_property_values(property_id, qcodes)
        except KeyError:
            # property does not exist
            property_vals = ["DONTEXIST" for item in qcodes]

        # bool_matches is a list of booleans, describing whether the query value is either equal to the
        # value for the Wikidata property (if string) or in the list of values for that Wikidata property
        # (if list)
        bool_matches = []

        for p in property_vals:
            if isinstance(p, str):
                bool_matches.append(p == match_val)
            elif isinstance(p, list):
                bool_matches.append(match_val in p)

        return list(compress(qcodes, bool_matches))

    def _apply_label_filter(
        self, qcode_col: str, filter_params: list, row: pd.Series
    ) -> list:
        """
        Apply a label filter to a row. Uses fuzzy matching to filter IDs to ones whose labels are similar to
        the specified dataframe column. 

        Args:
            qcode_col (str): the column name in the row to apply the filter to
            filter_params (list)
            row (pd.Series)

        Returns:
            list: list of qcodes after applying filter
        """
        # TODO: enable include_aliases
        # get filter params and qcodes
        string_col, threshold, include_aliases = filter_params
        qcodes = row[qcode_col]

        # create list of qcodes filtered on whether there were matches
        source_string = row[string_col]
        target_strings = self.entities.get_labels(qcodes)
        target_strings = (
            [target_strings] if isinstance(target_strings, str) else target_strings
        )

        def get_bool_matches(source_string, target_strings):
            return [
                fuzzy_match(source_string, t, threshold=threshold)
                for t in target_strings
            ]

        if include_aliases:
            aliases = self.entities.get_aliases(qcodes)

            if len(qcodes) == 1:
                # only one item to look up -> add aliases onto target_strings
                target_strings += aliases
                string_matches = get_bool_matches(source_string, target_strings)

                # if any of the strings matched, return qcodes
                if any(x for x in string_matches):
                    return qcodes
                else:
                    return []

            else:
                target_strings = [
                    aliases[i] + [target_strings[i]]
                    for i in range(0, len(target_strings))
                ]
                bool_matches = []

                # for each list in the list of lists, find out whether there are any matches
                for item in target_strings:
                    string_matches = get_bool_matches(source_string, item)
                    bool_matches.append(any(x for x in string_matches))

                return list(compress(qcodes, bool_matches))

        else:
            bool_matches = get_bool_matches(source_string, target_strings)

        # print(source_string, target_strings, bool_matches)

        return list(compress(qcodes, bool_matches))

    def process_dataframe(self) -> pd.DataFrame:
        """
        Processes dataframe with filter created and returns the processed dataframe.
        Also shows a message of how many have matched.

        Returns:
            pd.DataFrame: [description]
        """

        self.df.loc[:, self.new_qcode_col] = self.df.loc[:, self.qcode_col]

        i = 0

        for k, params in self.filters.items():
            print(f"Processing filter {i+1} of {len(self.filters)}")
            # filter what we process down to rows where we still have qcodes
            df_to_process = self.df[
                self.df[self.new_qcode_col].map(lambda d: len(d)) > 0
            ]

            if k == "property":
                for idx, row in tqdm(
                    df_to_process.iterrows(), total=df_to_process.shape[0]
                ):
                    self.df.at[idx, self.new_qcode_col] = self._apply_property_filter(
                        self.new_qcode_col, params, row
                    )

            elif k == "label":
                for idx, row in tqdm(
                    df_to_process.iterrows(), total=df_to_process.shape[0]
                ):
                    self.df.at[idx, self.new_qcode_col] = self._apply_label_filter(
                        self.new_qcode_col, params, row
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
