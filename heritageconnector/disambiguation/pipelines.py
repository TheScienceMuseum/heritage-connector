from elasticsearch import helpers
import rdflib
from rdflib import Graph
import json
import math
from itertools import islice
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from typing import Tuple
import time
import os
import csv
from joblib import dump, load

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
)
from heritageconnector.utils.generic import paginate_generator
from heritageconnector.namespace import OWL, RDF, RDFS
from heritageconnector.disambiguation.retrieve import get_wikidata_fields
from heritageconnector.disambiguation.search import es_text_search
from heritageconnector.disambiguation import compare_fields as compare
from heritageconnector import logging, errors

logger = logging.get_logger(__name__)


class Disambiguator(Classifier):
    def __init__(
        self,
        random_state=42,
        max_depth=5,
        class_weight="balanced",
        min_samples_split=2,
        min_samples_leaf=5,
        max_features=None,
    ):
        super().__init__()

        self.clf = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

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

    def predict(self, X, threshold=0.5):
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
        self,
        path: str,
        table_name: str,
        limit: int = None,
        page_size=100,
        search_limit=20,
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
            table_name (str): table name of entities to process
            limit (int, optional): Optionally limit the number of records processed. Defaults to None.
            page_size (int, optional): Batch size. Defaults to 100.
            search_limit (int, optional): Number of Wikidata candidates to process per SMG record, one of which
                is the correct match. Defaults to 20.
        """

        if not os.path.exists(path):
            errors.raise_file_not_found_error(path, "folder")

        X, y, pid_labels, id_pairs = self.build_training_data(
            True,
            table_name,
            page_size=page_size,
            limit=limit,
            search_limit=search_limit,
        )

        np.save(os.path.join(path, "X.npy"), X)
        np.save(os.path.join(path, "y.npy"), y)

        with open(os.path.join(path, "pids.txt"), "w") as f:
            f.write("\n".join(pid_labels))

        with open(os.path.join(path, "ids.txt"), "w") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerows(id_pairs)

    def save_test_data_to_folder(
        self,
        path: str,
        table_name: str,
        limit: int = None,
        page_size=100,
        search_limit=20,
    ):
        """
        Make test data from the unlabelled records in the Heritage Connector and save it to a folder. The folder will contain:
            - X.npy: numpy array X
            - pids.txt: newline separated list of column labels of X (properties used)
            - ids.txt: tab-separated CSV (tsv) of internal and external ID pairs (rows of X)

        These can be loaded from the folder using `heritageconnector.disambiguation.helpers.load_training_data`.

        Args:
            path (str): path of folder to save files to
            table_name (str): table name of entities to process
            limit (int, optional): Optionally limit the number of records processed. Defaults to None.
            page_size (int, optional): Batch size. Defaults to 100.
            search_limit (int, optional): Number of Wikidata candidates to process per SMG record, one of which
                is the correct match. Defaults to 20.
        """

        if not os.path.exists(path):
            errors.raise_file_not_found_error(path, "folder")

        X, pid_labels, id_pairs = self.build_training_data(
            False,
            table_name,
            page_size=page_size,
            limit=limit,
            search_limit=search_limit,
        )

        np.save(os.path.join(path, "X.npy"), X)

        with open(os.path.join(path, "pids.txt"), "w") as f:
            f.write("\n".join(pid_labels))

        with open(os.path.join(path, "ids.txt"), "w") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerows(id_pairs)

    def _get_pids(
        self,
        table_name,
        ignore_pids=["description"],
        used_types=["numeric", "string", "categorical"],
    ) -> list:
        """
        Get an ordered list of PIDS that have been used for training (the column names of X).

        Returns:
            list
        """
        table_mapping = field_mapping.mapping[table_name]

        pids = []

        for _, v in table_mapping.items():
            if (
                (("PID" in v) or (v.get("RDF") == RDFS.label))
                and ("RDF" in v)
                and (v.get("PID") not in ignore_pids)
                and (v.get("type") in used_types)
            ):
                if "PID" in v:
                    pids.append(url_to_pid(v["PID"]))
                elif v.get("RDF") == RDFS.label:
                    pids.append("label")

        return pids

    def _process_wikidata_results(self, wikidata_results: pd.DataFrame) -> pd.DataFrame:
        """
        - fill empty firstname (P735) and lastname (P734) fields by taking the first and last words of the label field
        - convert birthdate & deathdate (P569 & P570) to years
        - add label column combining itemLabel and altLabel lists
        """
        firstname_from_label = lambda l: l.split(" ")[0]
        lastname_from_label = lambda l: l.split(" ")[-1]
        year_from_wiki_date = (
            # don't worry about converting to numeric type here as comparison functions handle this
            lambda l: l[1:5]
            if isinstance(l, str)
            else [i[1:5] for i in l]
        )

        # firstname, lastname
        if "P735" in wikidata_results.columns and "P734" in wikidata_results.columns:
            for idx, row in wikidata_results.iterrows():
                wikidata_results.loc[idx, "P735"] = (
                    firstname_from_label(row["label"])
                    if not row["P735"]
                    else row["P735"]
                )
                wikidata_results.loc[idx, "P734"] = (
                    lastname_from_label(row["label"])
                    if not row["P734"]
                    else row["P734"]
                )

        # date of birth, date of death
        if "P569" in wikidata_results.columns and "P570" in wikidata_results.columns:
            wikidata_results["P569"] = wikidata_results["P569"].apply(
                year_from_wiki_date
            )
            wikidata_results["P570"] = wikidata_results["P570"].apply(
                year_from_wiki_date
            )

        if "P571" in wikidata_results.columns and "P576" in wikidata_results.columns:
            wikidata_results["P571"] = wikidata_results["P571"].apply(
                year_from_wiki_date
            )
            wikidata_results["P576"] = wikidata_results["P576"].apply(
                year_from_wiki_date
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

    def _get_labelled_records_from_elasticsearch(
        self, table_name: str, limit: int = None
    ):
        """
        Get labelled records (with sameAs) from Elasticsearch for training.

        Args:
            table_name (str):
            limit (int, optional): Defaults to None.

        """

        query = {
            "query": {
                "bool": {
                    "must": [
                        {"wildcard": {"graph.@owl:sameAs.@id": "*"}},
                        {"term": {"type.keyword": table_name.upper()}},
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

    def _get_unlabelled_records_from_elasticsearch(
        self, table_name: str, limit: int = None
    ):
        """
        Get unlabelled records (without sameAs) from Elasticsearch for inference.

        Args:
            table_name (str)
            limit (int, optional): Defaults to None.
        """

        query = {
            "query": {
                "bool": {
                    "must": {"term": {"type.keyword": table_name.upper()}},
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
                    ] = get_distance_between_entities_multiple(ent_set, reciprocal=True)

    def build_training_data(
        self,
        train: bool,
        table_name: str,
        page_size: int = 100,
        limit: int = None,
        search_limit=20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training arrays X, y from all the records in the Heritage Connector index with an existing sameAs
        link to Wikidata.

        Args:
            wd_index (str): Elasticsearch index of the Wikidata dump
            train (str): whether to build training data (True) or data for inference (False). If True a y vector
                is returned, otherwise one isn't.
            table_name (str): table name in field_mapping config
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
        table_mapping = field_mapping.mapping[table_name]
        wd_index = config.ELASTIC_SEARCH_WIKI_INDEX
        search = es_text_search(index=wd_index)

        filtered_mapping = {
            k: v
            for (k, v) in table_mapping.items()
            if (("PID" in v) or (v.get("RDF") == RDFS.label))
            and ("RDF" in v)
            and (v.get("PID") != "description")
        }
        pids = [
            url_to_pid(v["PID"])
            for _, v in filtered_mapping.items()
            if v["RDF"] != RDFS.label
        ]
        # also get URI of instanceof property (P31)
        pids_nolabel = [
            url_to_pid(v["PID"])
            for _, v in table_mapping.items()
            if v.get("wikidata_entity")
        ] + ["P31"]
        X_list = []
        if train:
            y_list = []
        ent_similarity_list = []
        id_pair_list = []

        # get records to process from Elasticsearch
        if train:
            search_res = self._get_labelled_records_from_elasticsearch(
                table_name, limit
            )
        else:
            search_res = self._get_unlabelled_records_from_elasticsearch(
                table_name, limit
            )

        search_res_paginated = paginate_generator(search_res, page_size)

        total = math.ceil(limit / page_size) if limit is not None else None

        # for each record, get Wikidata results and create X: feature matrix and y: boolean vector (correct/incorrect match)
        for item_list in tqdm(search_res_paginated, total=total):
            id_qid_mapping = dict()
            qid_instanceof_mapping = dict()
            batch_instanceof_comparisons = []
            # below is used so graphs can be accessed between the first and second `for item in item_list` loop
            graph_list = []

            logger.debug("Running search")
            start = time.time()
            for item in item_list:
                g = Graph().parse(
                    data=json.dumps(item["_source"]["graph"]), format="json-ld"
                )
                graph_list.append(g)
                item_id = next(g.subjects())
                item_label = g.label(next(g.subjects()))

                # search for Wikidata matches and retrieve information (batched)
                qids, qid_instanceof_temp = search.run_search(
                    item_label,
                    limit=search_limit,
                    include_aliases=True,
                    return_instanceof=True,
                )
                id_qid_mapping[item_id] = qids
                qid_instanceof_mapping.update(qid_instanceof_temp)
            end = time.time()
            logger.debug(f"...search complete in {end-start}s")

            # get Wikidata property values for the batch
            logger.debug("Getting wikidata fields")
            start = time.time()
            wikidata_results_df = get_wikidata_fields(
                pids=pids, id_qid_mapping=id_qid_mapping, pids_nolabel=pids_nolabel
            )
            end = time.time()
            logger.debug(f"...retrieved in {end-start}s")

            wikidata_results_df = self._process_wikidata_results(wikidata_results_df)

            # create X array for each record
            for idx, item in enumerate(item_list):
                X_temp = []
                g = graph_list[idx]
                item_id = next(g.subjects())
                qids_wikidata = wikidata_results_df.loc[
                    wikidata_results_df["id"] == item_id, "qid"
                ]
                if train:
                    item_qid = url_to_qid(next(g.objects(predicate=OWL.sameAs)))
                    y_item = [item_qid == qid for qid in qids_wikidata]
                id_pairs = [[str(item_id), qid] for qid in qids_wikidata]

                # calculate instanceof distances
                try:
                    item_instanceof = url_to_qid(next(g.objects(predicate=RDF.type)))
                    wikidata_instanceof = wikidata_results_df.loc[
                        wikidata_results_df["id"] == item_id, "P31"
                    ].tolist()

                    to_tuple = lambda x: tuple(x) if isinstance(x, list) else x
                    batch_instanceof_comparisons += [
                        (item_instanceof, to_tuple(url_to_qid(q, raise_invalid=False)))
                        for q in wikidata_instanceof
                    ]
                except:  # noqa: E722
                    logger.warning("Getting types for comparison failed.")

                    batch_instanceof_comparisons += [
                        (None, None)
                        for q in range(
                            len(
                                wikidata_results_df.loc[
                                    wikidata_results_df["id"] == item_id, :
                                ]
                            )
                        )
                    ]

                for key, value in filtered_mapping.items():
                    pid = (
                        "label"
                        if value["RDF"] == RDFS.label
                        else url_to_pid(value["PID"])
                    )
                    rdf = value["RDF"]
                    val_type = value["type"]

                    # TODO: is there a better way than storing these graphs in memory then retrieving parts of the graph here?
                    vals_internal = [str(i) for i in g.objects(predicate=rdf)]
                    vals_wikidata = wikidata_results_df.loc[
                        wikidata_results_df["id"] == item_id, pid
                    ].tolist()
                    # convert Wikidata QIDs to URL so they can be compared against RDF predicates of internal DB
                    vals_wikidata = [
                        qid_to_url(i) if is_qid(i) else i for i in vals_wikidata
                    ]

                    if val_type == "string":
                        sim_list = [
                            compare.similarity_string(vals_internal, i)
                            for i in vals_wikidata
                        ]

                    elif val_type == "numeric":
                        sim_list = [
                            compare.similarity_numeric(vals_internal, i)
                            if str(i) != ""
                            else 0
                            for i in vals_wikidata
                        ]

                    elif val_type == "categorical":
                        sim_list = [
                            compare.similarity_categorical(
                                vals_internal, i, raise_on_diff_types=False
                            )
                            for i in vals_wikidata
                        ]

                    if val_type in ["string", "numeric", "categorical"]:
                        X_temp.append(sim_list)

                X_item = np.asarray(X_temp, dtype=np.float32).transpose()
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
            X_columns = self._get_pids(table_name) + ["P31"]

            return X, y, X_columns, id_pair_list

        else:
            X = np.column_stack([np.vstack(X_list), ent_similarity_list])
            X_columns = self._get_pids(table_name) + ["P31"]

            return X, X_columns, id_pair_list
