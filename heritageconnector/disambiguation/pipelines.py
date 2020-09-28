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

from heritageconnector.datastore import es
from heritageconnector.config import config, field_mapping
from heritageconnector.utils.wikidata import (
    url_to_pid,
    url_to_qid,
    get_distance_between_entities_multiple,
)
from heritageconnector.utils.generic import paginate_generator
from heritageconnector.namespace import OWL, RDF, RDFS
from heritageconnector.disambiguation.retrieve import get_wikidata_fields
from heritageconnector.disambiguation.search import es_text_search
from heritageconnector.disambiguation import compare_fields as compare
from heritageconnector import logging

logger = logging.get_logger(__name__)


def get_pids(
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


def _process_wikidata_results(wikidata_results: pd.DataFrame) -> pd.DataFrame:
    """
    - fill empty firstname (P735) and lastname (P734) fields by taking the first and last words of the label field
    - convert birthdate & deathdate (P569 & P570) to years
    - add label column combining itemLabel and altLabel lists
    """
    firstname_from_label = lambda l: l.split(" ")[0]
    lastname_from_label = lambda l: l.split(" ")[-1]
    year_from_wiki_date = (
        # don't worry about converting to numeric type here as comparison functions handle this
        lambda l: l[0:4]
        if isinstance(l, str)
        else [i[0:4] for i in l]
    )

    # firstname, lastname
    if "P735" in wikidata_results.columns and "P734" in wikidata_results.columns:
        for idx, row in wikidata_results.iterrows():
            wikidata_results.loc[idx, "P735"] = (
                firstname_from_label(row["itemLabel"])
                if not row["P735"]
                else row["P735"]
            )
            wikidata_results.loc[idx, "P734"] = (
                lastname_from_label(row["itemLabel"])
                if not row["P734"]
                else row["P734"]
            )

    # date of birth, date of death
    if "P569" in wikidata_results.columns and "P570" in wikidata_results.columns:
        wikidata_results["P569"] = wikidata_results["P569"].apply(year_from_wiki_date)
        wikidata_results["P570"] = wikidata_results["P570"].apply(year_from_wiki_date)

    if "P571" in wikidata_results.columns and "P576" in wikidata_results.columns:
        wikidata_results["P571"] = wikidata_results["P571"].apply(year_from_wiki_date)
        wikidata_results["P576"] = wikidata_results["P576"].apply(year_from_wiki_date)

    # combine labels and aliases into one list: label
    wikidata_results["itemLabel"] = wikidata_results["itemLabel"].apply(
        lambda i: [i] if isinstance(i, str) else i
    )
    wikidata_results["altLabel"] = wikidata_results["altLabel"].apply(
        lambda i: [i] if isinstance(i, str) else i
    )
    wikidata_results["label"] = (
        wikidata_results["itemLabel"] + wikidata_results["altLabel"]
    )

    return wikidata_results


def build_training_data(
    table_name: str, page_size: int = 100, limit: int = None, search_limit=20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get training arrays X, y from all the records in the Heritage Connector index with an existing sameAs
    link to Wikidata.

    Args:
        wd_index (str): Elasticsearch index of the Wikidata dump
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
    wd_index = field_mapping.wikidump_index
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
    y_list = []
    ent_similarity_list = []
    # in-memory caching for entity similarities, prefilled with case for where there is no type specified
    ent_similarities_lookup = {hash((None, None)): 0}
    id_pair_list = []

    # get records with sameAs from Elasticsearch
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
    # set timeout to longer than default here to deal with large times between subsequent ES requests
    search_res = helpers.scan(
        es, query=query, index=config.ELASTIC_SEARCH_INDEX, size=500, scroll="30m"
    )
    if limit:
        search_res = islice(search_res, limit)

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

        wikidata_results_df = _process_wikidata_results(wikidata_results_df)

        # create X array for each record
        for idx, item in enumerate(item_list):
            X_temp = []
            g = graph_list[idx]
            item_id = next(g.subjects())
            item_qid = url_to_qid(next(g.objects(predicate=OWL.sameAs)))
            qids_wikidata = wikidata_results_df.loc[
                wikidata_results_df["id"] == item_id, "item"
            ]
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
                    "label" if value["RDF"] == RDFS.label else url_to_pid(value["PID"])
                )
                rdf = value["RDF"]
                val_type = value["type"]

                # TODO: is there a better way than storing these graphs in memory then retrieving parts of the graph here?
                vals_internal = [str(i) for i in g.objects(predicate=rdf)]
                vals_wikidata = wikidata_results_df.loc[
                    wikidata_results_df["id"] == item_id, pid
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
            y_list += y_item
            id_pair_list += id_pairs

        batch_instanceof_comparisons_unique = list(set(batch_instanceof_comparisons))

        logger.debug("Finding distances between entities...")
        for ent_1, ent_2 in tqdm(batch_instanceof_comparisons_unique):
            if (ent_1, ent_2) != (None, None):
                if isinstance(ent_2, list):
                    ent_set = {ent_1, tuple(ent_2)}
                else:
                    ent_set = {ent_1, ent_2}

                if hash((ent_1, ent_2)) not in ent_similarities_lookup:
                    ent_similarities_lookup[
                        hash((ent_1, ent_2))
                    ] = get_distance_between_entities_multiple(ent_set, reciprocal=True)

        for ent_1, ent_2 in batch_instanceof_comparisons:
            ent_similarity_list.append(ent_similarities_lookup[hash((ent_1, ent_2))])

    X = np.column_stack([np.vstack(X_list), ent_similarity_list])
    y = np.asarray(y_list, dtype=bool)
    X_columns = get_pids(table_name) + ["P31"]

    return X, y, X_columns, id_pair_list
