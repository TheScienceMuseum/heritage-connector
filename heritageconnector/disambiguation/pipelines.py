from elasticsearch import helpers
import json
import math
from itertools import islice
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from typing import Tuple, List, Union, Iterable
import time
import os
import csv
from joblib import dump, load

import rdflib
from rdflib import Graph, URIRef
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

from heritageconnector.base.disambiguation import Classifier
from heritageconnector.datastore import es
from heritageconnector.config import config, field_mapping
from heritageconnector.utils.wikidata import (
    url_to_pid,
    url_to_qid,
    get_distance_between_entities_multiple,
    qid_to_url,
    is_qid,
    get_wikidata_equivalents_for_properties,
    filter_qids_in_class_tree,
)
from heritageconnector.utils.generic import paginate_generator
from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.namespace import OWL, RDF, RDFS, SKOS, FOAF
from heritageconnector.disambiguation.retrieve import get_wikidata_fields
from heritageconnector.disambiguation.search import es_text_search
from heritageconnector.disambiguation.compare_fields import (
    compare,
    similarity_categorical,
    similarity_string,
)
from heritageconnector import logging, errors

logger = logging.get_logger(__name__)


class Disambiguator(Classifier):
    """
    Implementation of a classifier for finding sameAs links between items in the Heritage Connector and items on Wikidata.
    TODO: link to documentation on exactly how this works.

    Attributes:
        table_name (str): `skos:hasTopConcept` value to use for disambiguator. This should 
            have been set to refer to its original data source when importing data to the graph.
        random_state (int, optional): random state for all methods involving randomness. Defaults to 42.
        TODO: tune these decision tree params automatically when training the classifier.
        max_depth (int, optional): max depth of the decision tree classifier.
        class_weight (str, optional): See sklearn.tree.DecisionTreeClassifier docs. Defaults to "balanced".
        min_samples_split (int, optional): See sklearn.tree.DecisionTreeClassifier docs. Defaults to 2.
        min_samples_leaf (int, optional): See sklearn.tree.DecisionTreeClassifier docs. Defaults to 5.
        max_features (int, optional): See sklearn.tree.DecisionTreeClassifier docs. Defaults to None.
        bidirectional_distance (bool, optional): whether to include Wikidata types not in the immediate 
            class tree when calculating similarity between entity types. Defaults to False, i.e. only considers 
            types to have a similarity greater than 0 if they are in the same instance of/subclass of Wikidata 
            hierarchy.
        enforce_entities_have_type (bool, optional): only entities with values for `rdf:type` will be retrieved
            from the heritage connector graph. Defaults to True.
    """

    def __init__(
        self,
        table_name: str,
        random_state=42,
        max_depth=5,
        class_weight="balanced",
        min_samples_split=2,
        min_samples_leaf=5,
        max_features=None,
        bidirectional_distance=False,
        enforce_entities_have_type=True,
    ):
        super().__init__()

        self.table_name = table_name.upper()
        self.table_mapping = field_mapping.mapping[self.table_name]
        self.enforce_entities_have_type = enforce_entities_have_type

        self.clf = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

        # whether to use an entity distance measure that can change direction
        self.bidirectional_distance = bidirectional_distance

        # in-memory caching for entity similarities, prefilled with case for where there is no type specified
        self.entity_distance_cache = {hash((None, None)): 0}

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf = self.clf.fit(X, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probabilities for the positive class

        Args:
            X (np.ndarray)

        Returns:
            np.ndarray: a value for each row of X
        """
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold=0.5) -> np.ndarray:
        """
        Returns predictions for the positive class at a threshold.

        Args:
            X (np.ndarray)
            threshold (float, optional): Defaults to 0.5.

        Returns:
            np.ndarray: boolean values
        """
        pred_proba = self.predict_proba(X)

        return pred_proba >= threshold

    def predict_top_ranked_pairs(
        self, X: np.ndarray, pairs: pd.DataFrame, threshold=0.5
    ) -> pd.DataFrame:
        """
        Returns a dataframe of highest ranked Wikidata candidate for each internal record based on the classifier output.
        Any predictions below the threshold aren't counted. If there are multiple Wikidata candidates with the same
        predicted probability, all candidates with the maximum probability are returned.

        Args:
            X (np.ndarray)
            pairs (pd.DataFrame): with columns internal_id, wikidata_id. returned by self.build_training_data
            threshold (float, optional): Defaults to 0.5.

        Returns:
            pd.DataFrame: with columns internal_id, wikidata_id, y_pred, y_pred_proba
        """

        pairs_new = pairs.copy()

        y_pred_proba = self.predict_proba(X)
        pairs_new["y_pred_proba"] = y_pred_proba
        pairs_new["y_pred"] = y_pred_proba >= threshold

        pairs_true = pairs_new[pairs_new["y_pred"] == True]  # noqa: E712

        pairs_true_filtered = pd.DataFrame()

        for _id in pairs_true["internal_id"].unique().tolist():
            tempdf = pairs_true[pairs_true["internal_id"] == _id]
            max_proba = tempdf["y_pred_proba"].max()

            pairs_true_filtered = pairs_true_filtered.append(
                tempdf[tempdf["y_pred_proba"] == max_proba]
            )

        return pairs_true_filtered

    def score(
        self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5, output_dict=False
    ) -> float:
        """
        Returns balanced accuracy, precision and recall for given test data and labels.

        Args:
            X (np.ndarray): data to return score for.
            y (np.ndarray): True labels.
            threshold (np.ndarray): threshold to use for classification.
            output_dict (bool, optional): whether to output a dictionary with the results. Defaults to False,
                where the results will be printed.

        Returns:
            float: score
        """

        y_pred = self.predict(X, threshold)

        results = {
            "balanced accuracy score": balanced_accuracy_score(y, y_pred),
            "precision score": precision_score(y, y_pred),
            "recall score": recall_score(y, y_pred),
        }

        if output_dict:
            return results
        else:
            return "\n".join([f"{k}: {v}" for k, v in results.items()])

    def print_tree(self, feature_names: list = None):
        """
        Print textual representation of the decision tree.

        Args:
            feature_names (list, optional): List of feature names to use. Defaults to None.
        """

        print(export_text(self.clf, feature_names=feature_names))

    def save_classifier_to_disk(self, path: str):
        """
        Pickle classifier to disk.

        Args:
            path (str): path to pickle to
        """

        # TODO: should maybe raise a warning if model hasn't been trained,
        # but not sure how to do this without testing predict (which needs X, or
        # at least the required dimensions of X)

        dump(self.clf, path)

    def load_classifier_from_disk(self, path: str):
        """
        Load pickled classifier from disk

        Args:
            path (str): path of pickled classifier
        """

        # TODO: maybe there should be a warning if overwriting a trained model.
        # See todo above.

        self.clf = load(path)

    def save_training_data_to_folder(
        self, path: str, limit: int = None, page_size=100, search_limit=20,
    ):
        """
        Make training data from the labelled records in the Heritage Connector and save it to a folder. The folder will contain:
            - X.npy: numpy array X
            - y.npy: numpy array y
            - pids.txt: newline separated list of column labels of X (properties used)
            - ids.txt: tab-separated CSV (tsv) of internal and external ID pairs (rows of X)

        These can be loaded from the folder using `heritageconnector.disambiguation.helpers.load_training_data`.

        Args:
            path (str): path of folder to save files to
            limit (int, optional): Optionally limit the number of records processed. Defaults to None.
            page_size (int, optional): Batch size. Defaults to 100.
            search_limit (int, optional): Number of Wikidata candidates to process per SMG record, one of which
                is the correct match. Defaults to 20.
        """

        if not os.path.exists(path):
            errors.raise_file_not_found_error(path, "folder")

        X, y, pid_labels, id_pairs = self.build_training_data(
            True, page_size=page_size, limit=limit, search_limit=search_limit,
        )

        np.save(os.path.join(path, "X.npy"), X)
        np.save(os.path.join(path, "y.npy"), y)

        with open(os.path.join(path, "pids.txt"), "w") as f:
            f.write("\n".join(pid_labels))

        with open(os.path.join(path, "ids.txt"), "w") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerows(id_pairs)

    def save_test_data_to_folder(
        self, path: str, limit: int = None, page_size=100, search_limit=20,
    ):
        """
        Make test data from the unlabelled records in the Heritage Connector and save it to a folder. The folder will contain:
            - X.npy: numpy array X
            - pids.txt: newline separated list of column labels of X (properties used)
            - ids.txt: tab-separated CSV (tsv) of internal and external ID pairs (rows of X)

        These can be loaded from the folder using `heritageconnector.disambiguation.helpers.load_training_data`.

        Args:
            path (str): path of folder to save files to
            limit (int, optional): Optionally limit the number of records processed. Defaults to None.
            page_size (int, optional): Batch size. Defaults to 100.
            search_limit (int, optional): Number of Wikidata candidates to process per SMG record, one of which
                is the correct match. Defaults to 20.
        """

        if not os.path.exists(path):
            errors.raise_file_not_found_error(path, "folder")

        X, pid_labels, id_pairs = self.build_training_data(
            False, page_size=page_size, limit=limit, search_limit=search_limit,
        )

        np.save(os.path.join(path, "X.npy"), X)

        with open(os.path.join(path, "pids.txt"), "w") as f:
            f.write("\n".join(pid_labels))

        with open(os.path.join(path, "ids.txt"), "w") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerows(id_pairs)

    def _process_wikidata_results(self, wikidata_results: pd.DataFrame) -> pd.DataFrame:
        """
        - fill empty firstname (P735) and lastname (P734) fields by taking the first and last words of the label field
        - convert any date-like values to positive or negative integers
        - add label column combining itemLabel and altLabel lists
        """
        firstname_from_label = lambda l: l.split(" ")[0]
        lastname_from_label = lambda l: l.split(" ")[-1]

        # firstname, lastname
        if (
            "P735Label" in wikidata_results.columns
            and "P734Label" in wikidata_results.columns
        ):
            for idx, row in wikidata_results.iterrows():
                wikidata_results.loc[idx, "P735Label"] = (
                    firstname_from_label(row["label"])
                    if not row["P735Label"]
                    else row["P735Label"]
                )
                wikidata_results.loc[idx, "P734Label"] = (
                    lastname_from_label(row["label"])
                    if not row["P734Label"]
                    else row["P734Label"]
                )

        # combine labels and aliases into one list: label
        wikidata_results["label"] = wikidata_results["label"].apply(
            lambda i: [i] if isinstance(i, str) else i
        )
        wikidata_results["aliases"] = wikidata_results["aliases"].apply(
            lambda i: [i] if isinstance(i, str) else i
        )
        wikidata_results["label"] = (
            wikidata_results["label"] + wikidata_results["aliases"]
        )

        return wikidata_results

    def _get_geographic_properties(self, pids: List[str]) -> List[str]:
        """
        Filter list of properties to ones which are geographic properties. Used so
        they can be compared using a separate similarity function.

        Args:
            pids (list): Wikidata properties

        Returns:
            list: geographic properties only
        """

        # Q18615777 is 'Wikidata property to indicate a location'
        return filter_qids_in_class_tree(pids, "Q18615777", include_instanceof=True)

    def _get_labelled_records_from_elasticsearch(self, limit: int = None):
        """
        Get labelled records (with sameAs) from Elasticsearch for training.

        Args:
            limit (int, optional): Defaults to None.

        """

        query = {
            "query": {
                "bool": {
                    "must": [
                        {"wildcard": {"graph.@owl:sameAs.@id": "*"}},
                        {"term": {"type.keyword": self.table_name.upper()}},
                    ]
                }
            }
        }
        # set 'scroll' timeout to longer than default here to deal with large times between subsequent ES requests
        search_res = helpers.scan(
            es, query=query, index=config.ELASTIC_SEARCH_INDEX, size=500, scroll="30m"
        )
        if limit:
            search_res = islice(search_res, limit)

        return search_res

    def _get_unlabelled_records_from_elasticsearch(self, limit: int = None):
        """
        Get unlabelled records (without sameAs) from Elasticsearch for inference.

        Args:
            limit (int, optional): Defaults to None.
        """

        query = {
            "query": {
                "bool": {
                    "must": {"term": {"type.keyword": self.table_name.upper()}},
                    "must_not": {"exists": {"field": "graph.@owl:sameAs.@id"}},
                }
            }
        }

        search_res = helpers.scan(
            es, query=query, index=config.ELASTIC_SEARCH_INDEX, size=500, scroll="30m"
        )
        if limit:
            search_res = islice(search_res, limit)

        return search_res

    def _get_type_constraint(self) -> str:
        """For _get_labelled_records_from_sparql_store/_get_unlabelled_records_from_sparql_store"""

        if self.enforce_entities_have_type:
            return "?item rdf:type ?type."
        else:
            return ""

    def _get_labelled_records_from_sparql_store(
        self, limit: int = None
    ) -> Iterable[dict]:
        """
        Get all records with an owl:sameAs value (URIs and labels) from the Fuseki instance.

        Args:
            limit (int, optional): Defaults to None.

        Returns:
            Generator of dicts. Each dict has the form {"id": __, "label": ___}
        """

        query = f"""SELECT DISTINCT ?item ?itemLabel WHERE {{
            ?item owl:sameAs ?object.
            ?item rdfs:label ?itemLabel.
            {self._get_type_constraint()}
            ?item skos:hasTopConcept '{self.table_name}'.
        }}"""

        if limit is not None:
            query = query + f"LIMIT {limit}"

        res = get_sparql_results(config.FUSEKI_ENDPOINT, query)["results"]["bindings"]

        return (
            {"id": item["item"]["value"], "label": item["itemLabel"]["value"]}
            for item in res
        )

    def _get_unlabelled_records_from_sparql_store(
        self, limit: int = None
    ) -> Iterable[dict]:
        """
        Get all records without an owl:sameAs value (URIs and labels) from the Fuseki instance.

        Args:
            limit (int, optional): Defaults to None.

        Returns:
            Generator of dicts. Each dict has the form {"id": __, "label": ___}
        """

        query = f"""SELECT DISTINCT ?item ?itemLabel WHERE {{
            FILTER NOT EXISTS {{?item owl:sameAs ?object}}.
            ?item rdfs:label ?itemLabel.
            {self._get_type_constraint()}
            ?item skos:hasTopConcept '{self.table_name}'.
        }}"""

        if limit is not None:
            query = query + f"LIMIT {limit}"

        res = get_sparql_results(config.FUSEKI_ENDPOINT, query)["results"]["bindings"]

        return (
            {"id": item["item"]["value"], "label": item["itemLabel"]["value"]}
            for item in res
        )

    def _get_predicates(
        self,
        predicates_ignore: List[str] = [
            RDFS.label,
            OWL.sameAs,
            SKOS.hasTopConcept,
            FOAF.title,
        ],
    ) -> List[str]:
        """
        Get a unique list of predicates for the table. These will form the columns of X.

        Args:
            predicates_ignore (List[str]): predicates to ignore

        Returns:
            list of URLs for each predicate, excluding those in `predicates_ignore`
        """

        # TODO: remove this when using pydantic as it will coerce rdflib.term.URIRef to string
        predicates_ignore = [str(i) for i in predicates_ignore]

        query = f"""
        SELECT DISTINCT ?predicate
        WHERE {{
        ?subject <http://www.w3.org/2004/02/skos/core#hasTopConcept> '{self.table_name}'.
        ?subject ?predicate ?object.
        }}"""

        res = get_sparql_results(config.FUSEKI_ENDPOINT, query)["results"]["bindings"]

        if len(res) > 0:
            return [
                i["predicate"]["value"]
                for i in res
                if i["predicate"]["value"] not in predicates_ignore
            ]

        else:
            return []

    def _open_sparql_store(self, endpoint: str = config.FUSEKI_ENDPOINT):
        """
        Open RDFlib SPARQL store with query URL at `endpoint`.

        Args:
            endpoint (str, optional): Defaults to config.FUSEKI_ENDPOINT.
        """

        self.sparql_store = SPARQLStore(endpoint)
        self.sparql_store.open(endpoint)

    def _get_triples_from_store(
        self, spo: tuple = (None, None, None)
    ) -> Iterable[tuple]:
        """
        Get triples with the mask (subject, predicate, object). Returns generator of tuples, where
        each tuple is a triple (ignores graph names).

        By default the SPARQL store is at the endpoint specified by FUSEKI_ENDPOINT in config. If you want
        to change this, call `self._open_sparql_store(endpoint='http://my_endpoint')` first.
        """
        if not hasattr(self, "sparqlstore"):
            self._open_sparql_store()

        return self.sparql_store.triples(spo)

    def _add_instanceof_distances_to_inmemory_cache(self, batch_instanceof_comparisons):
        """
        Adds instanceof distances for a batch to the in-memory/in-class-instance cache.
        """

        batch_instanceof_comparisons_unique = list(set(batch_instanceof_comparisons))

        logger.debug("Finding distances between entities...")
        for ent_1, ent_2 in tqdm(batch_instanceof_comparisons_unique):
            if (ent_1, ent_2) != (None, None):
                if isinstance(ent_2, list):
                    ent_set = {ent_1, tuple(ent_2)}
                else:
                    ent_set = {ent_1, ent_2}

                if hash((ent_1, ent_2)) not in self.entity_distance_cache:
                    self.entity_distance_cache[
                        hash((ent_1, ent_2))
                    ] = get_distance_between_entities_multiple(
                        ent_set,
                        bidirectional=self.bidirectional_distance,
                        reciprocal=True,
                    )

    def _to_tuple(self, val):
        """Convert lists to tuples, but leave values that aren't lists as they are."""
        return tuple(val) if isinstance(val, list) else val

    def build_training_data(
        self, train: bool, page_size: int = 100, limit: int = None, search_limit=20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training arrays X, y from all the records in the Heritage Connector index with an existing sameAs
        link to Wikidata.

        Args:
            train (str): whether to build training data (True) or data for inference (False). If True a y vector
                is returned, otherwise one isn't.
            page_size (int, optional): the number of records to fetch from Wikidata per iteration. Larger numbers
                will speed up the process but may cause the SPARQL query to time out. Defaults to 10.
                (TODO: set better default)
            limit (int, optional): set a limit on the number of records to use for training (useful for testing).
                Defaults to None.
            search_limit (int, optional): number of search results to retrieve from the Wikidata dump per record.
                Defaults to 20.

        Returns:
            Tuple[np.ndarray, np.ndarray]: X, y
        """

        predicates = self._get_predicates()
        predicate_pid_mapping = get_wikidata_equivalents_for_properties(predicates)
        pids_ignore = (config.PIDS_IGNORE).split(" ")
        pids_categorical = (config.PIDS_CATEGORICAL).split(" ")

        # remove instanceof (P31) and add to end, as the type distance calculations are appended to X last
        predicate_pid_mapping = {
            k: url_to_pid(v)
            for k, v in predicate_pid_mapping.items()
            if v is not None and url_to_pid(v) not in pids_ignore + ["P31"]
        }
        pids = list(predicate_pid_mapping.values()) + ["P31"]
        predicate_pid_mapping.update({RDFS.label: "label"})

        pids_geographical = self._get_geographic_properties(pids)

        X_list = []
        if train:
            y_list = []
        ent_similarity_list = []
        id_pair_list = []

        # get records to process from Elasticsearch
        search = es_text_search(index=config.ELASTIC_SEARCH_WIKI_INDEX)

        if train:
            search_res = self._get_labelled_records_from_sparql_store(limit)
        else:
            search_res = self._get_unlabelled_records_from_sparql_store(limit)

        search_res_paginated = paginate_generator(search_res, page_size)

        total = None if limit is None else math.ceil(limit / page_size)

        # for each record, get Wikidata results and create X: feature matrix and y: boolean vector (correct/incorrect match)
        for item_list in tqdm(search_res_paginated, total=total):
            id_qid_mapping = dict()
            qid_instanceof_mapping = dict()
            batch_instanceof_comparisons = []

            logger.debug("Running search")
            start = time.time()
            for item in item_list:
                # text search for Wikidata matches
                qids, qid_instanceof_temp = search.run_search(
                    item["label"],
                    limit=search_limit,
                    include_aliases=True,
                    return_instanceof=True,
                )
                id_qid_mapping[item["id"]] = qids
                qid_instanceof_mapping.update(qid_instanceof_temp)

            end = time.time()
            logger.debug(f"...search complete in {end-start}s")

            # get Wikidata property values for the batch
            logger.debug("Getting wikidata fields")
            start = time.time()
            wikidata_results_df = get_wikidata_fields(
                pids=pids, id_qid_mapping=id_qid_mapping
            )
            end = time.time()
            logger.debug(f"...retrieved in {end-start}s")

            wikidata_results_df = self._process_wikidata_results(wikidata_results_df)

            logger.debug("Calculating field similarities for batch..")
            # create X array for each record
            for item in item_list:
                # we get all the triples for the item here (rather than each triple in the for loop below)
                # to reduce the load on the SPARQL DB
                try:
                    item_triples = list(
                        self._get_triples_from_store((URIRef(item["id"]), None, None))
                    )

                except:  # noqa: E722
                    # sparql store has crashed
                    sleep_time = 120
                    logger.debug(
                        f"get_triples query failed. Retrying in {sleep_time} seconds"
                    )
                    time.sleep(sleep_time)
                    self._open_sparql_store()
                    item_triples = list(
                        self._get_triples_from_store((URIRef(item["id"]), None, None))
                    )

                X_temp = []
                qids_wikidata = wikidata_results_df.loc[
                    wikidata_results_df["id"] == item["id"], "qid"
                ]

                if train:
                    item_qid = url_to_qid(
                        [i for i in item_triples if i[0][1] == OWL.sameAs][0][0][-1]
                    )
                    y_item = [item_qid == qid for qid in qids_wikidata]

                id_pairs = [[item["id"], qid] for qid in qids_wikidata]

                # calculate instanceof distances
                try:
                    item_instanceof = [
                        url_to_qid(i[0][-1])
                        for i in item_triples
                        if i[0][1] == RDF.type
                    ]
                    wikidata_instanceof = wikidata_results_df.loc[
                        wikidata_results_df["id"] == item["id"], "P31"
                    ].tolist()

                    batch_instanceof_comparisons += [
                        (
                            self._to_tuple(item_instanceof),
                            self._to_tuple(url_to_qid(q, raise_invalid=False)),
                        )
                        for q in wikidata_instanceof
                    ]
                except:  # noqa: E722
                    # TODO: better error handling here. Why does this fail?
                    logger.warning("Getting types for comparison failed.")

                    batch_instanceof_comparisons += [
                        (None, None)
                        for q in range(
                            len(
                                wikidata_results_df.loc[
                                    wikidata_results_df["id"] == item["id"], :
                                ]
                            )
                        )
                    ]

                for predicate, pid in predicate_pid_mapping.items():
                    item_values = [
                        i for i in item_triples if i[0][1] == URIRef(predicate)
                    ]

                    # RDFS.label is a special case that has no associated PID. We just want to compare it
                    # to the 'label' column which is the labels + aliases for each Wikidata item.
                    if predicate == RDFS.label:
                        item_labels = [str(triple[0][-1]) for triple in item_values]
                        wikidata_labels = wikidata_results_df.loc[
                            wikidata_results_df["id"] == item["id"], "label"
                        ].tolist()
                        sim_list = [
                            similarity_string(item_labels, label_list)
                            for label_list in wikidata_labels
                        ]

                    elif pid in pids_geographical:
                        item_values = self._to_tuple(
                            url_to_qid(
                                [triple[0][-1] for triple in item_values],
                                raise_invalid=False,
                            )
                        )

                        wikidata_values = wikidata_results_df.loc[
                            wikidata_results_df["id"] == item["id"], pid
                        ].tolist()

                        if len(item_values) == 0:
                            sim_list = [1] * len(wikidata_values)
                        else:
                            sim_list = [
                                get_distance_between_entities_multiple(
                                    {self._to_tuple(wiki_val), item_values},
                                    vertex_pid="P131",
                                    reciprocal=True,
                                )
                                for wiki_val in wikidata_values
                            ]

                    else:
                        # TODO: if entity is a SMG entity, do we want to get its sameAs link or label?
                        wikidata_values = wikidata_results_df.loc[
                            wikidata_results_df["id"] == item["id"], pid
                        ].tolist()
                        wikidata_labels = wikidata_results_df.loc[
                            wikidata_results_df["id"] == item["id"], pid + "Label"
                        ].tolist()

                        if len(item_values) == 0:
                            # if the internal item has no values for the PID return zero similarity
                            # for this PID with each of the candidate QIDs
                            sim_list = [0] * len(wikidata_values)

                        else:
                            item_values = [triple[0][-1] for triple in item_values]
                            if pid in pids_categorical:
                                sim_list = [
                                    similarity_categorical(
                                        [str(i) for i in item_values],
                                        label,
                                        raise_on_diff_types=False,
                                    )
                                    for label in wikidata_labels
                                ]
                            else:
                                sim_list = [
                                    compare(
                                        item_values,
                                        wikidata_values[i],
                                        wikidata_labels[i],
                                    )
                                    for i in range(len(wikidata_values))
                                ]

                    X_temp.append(sim_list)

                X_item = np.asarray(X_temp, dtype=np.float32).transpose()

                # TODO (checkpoint): here we would want to save X_list, y_list, id_pair_list, self.entity_distance_cache to disk
                X_list.append(X_item)

                if train:
                    y_list += y_item

                id_pair_list += id_pairs

            self._add_instanceof_distances_to_inmemory_cache(
                batch_instanceof_comparisons
            )

            for ent_1, ent_2 in batch_instanceof_comparisons:
                ent_similarity_list.append(
                    self.entity_distance_cache[hash((ent_1, ent_2))]
                )

        if train:
            X = np.column_stack([np.vstack(X_list), ent_similarity_list])
            y = np.asarray(y_list, dtype=bool)
            X_columns = list(predicate_pid_mapping.values()) + ["P31"]

            return X, y, X_columns, id_pair_list

        else:
            X = np.column_stack([np.vstack(X_list), ent_similarity_list])
            X_columns = list(predicate_pid_mapping.values()) + ["P31"]

            return X, X_columns, id_pair_list
