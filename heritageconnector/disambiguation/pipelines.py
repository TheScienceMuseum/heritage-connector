from elasticsearch import helpers
import rdflib
from rdflib import Graph
import json
from itertools import islice
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from typing import Tuple

from heritageconnector.datastore import es
from heritageconnector.config import config, field_mapping
from heritageconnector.utils.wikidata import url_to_pid, url_to_qid
from heritageconnector.namespace import OWL, RDFS
from heritageconnector.disambiguation.retrieve import get_wikidata_fields
from heritageconnector.disambiguation.search import es_text_search
from heritageconnector.disambiguation import compare_fields as compare


def _process_wikidata_results(wikidata_results: pd.DataFrame) -> pd.DataFrame:
    """
    - fill empty firstname (P735) and lastname (P734) fields by taking the first and last words of the label field
    - convert birthdate & deathdate (P569 & P570) to years
    - add label column combining itemLabel and altLabel lists
    """
    firstname_from_label = lambda l: l.split(" ")[0]
    lastname_from_label = lambda l: l.split(" ")[-1]
    year_from_wiki_date = (
        lambda l: l[0:4] if isinstance(l, str) else np.mean([int(i[0:4]) for i in l])
    )

    # firstname, lastname
    for idx, row in wikidata_results.iterrows():
        wikidata_results.loc[idx, "P735"] = (
            firstname_from_label(row["itemLabel"]) if not row["P735"] else row["P735"]
        )
        wikidata_results.loc[idx, "P734"] = (
            lastname_from_label(row["itemLabel"]) if not row["P734"] else row["P734"]
        )

    # date of birth, date of death
    wikidata_results["P569"] = wikidata_results["P569"].apply(year_from_wiki_date)
    wikidata_results["P570"] = wikidata_results["P570"].apply(year_from_wiki_date)

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
    wd_index: str, table_name: str, limit: int = None, search_limit=20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get training arrays X, y from all the records in the Heritage Connector index with an existing sameAs
        link to Wikidata.

    Args:
        wd_index (str): Elasticsearch index of the Wikidata dump
        table_name (str): table name in field_mapping config
        limit (int, optional): set a limit on the number of records to use for training (useful for testing). 
            Defaults to None.
        search_limit (int, optional): number of search results to retrieve from the Wikidata dump per record. 
            Defaults to 20.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X, y
    """
    search = es_text_search(index=wd_index)
    table_mapping = field_mapping.mapping[table_name]

    filtered_mapping = {
        k: v
        for (k, v) in table_mapping.items()
        if (("PID" in v) or (v.get("RDF") == RDFS.label))
        and ("RDF" in v)
        and (v.get("PID") != "description")
    }
    pids_nolabel = [
        url_to_pid(v["PID"])
        for _, v in table_mapping.items()
        if v.get("wikidata_entity")
    ]
    X_list = []
    y_list = []

    # get records with sameAs from Elasticsearch
    query = {"query": {"wildcard": {"graph.@owl:sameAs.@id.keyword": "@wd*"}}}
    search_res = helpers.scan(es, query=query, index=config.ELASTIC_SEARCH_INDEX)
    if limit:
        search_res = islice(search_res, limit)

    # for each record, get Wikidata results and create X: feature matrix and y: boolean vector (correct/incorrect match)
    for item in tqdm(search_res, total=limit):
        g = Graph().parse(data=json.dumps(item["_source"]["graph"]), format="json-ld")
        label = g.label(next(g.subjects()))
        qid_true = url_to_qid(next(g.objects(predicate=OWL.sameAs)))
        X_temp = []

        # search for Wikidata matches and retrieve information (batched)
        # TODO: batch Wikidata query (multiple records per query)
        qids = search.run_search(label, limit=search_limit, include_aliases=True)
        pids = [
            url_to_pid(v["PID"])
            for _, v in filtered_mapping.items()
            if v["RDF"] != RDFS.label
        ]
        y_list += [qid == qid_true for qid in qids]

        # return and process table of results
        wikidata_results_df = get_wikidata_fields(
            pids=pids, qids=qids, pids_nolabel=pids_nolabel
        )
        wikidata_results_df = _process_wikidata_results(wikidata_results_df)

        # create vector for each record with True/False label according to whether there is a match
        for key, value in filtered_mapping.items():
            pid = "label" if value["RDF"] == RDFS.label else url_to_pid(value["PID"])

            rdf = value["RDF"]
            val_type = value["type"]

            vals_internal = [str(i) for i in g.objects(predicate=rdf)]
            vals_wikidata = wikidata_results_df.loc[
                wikidata_results_df["item"].isin(qids), pid
            ]

            if val_type == "string":
                sim_list = [
                    compare.similarity_string(vals_internal, i) for i in vals_wikidata
                ]

            elif val_type == "numeric":
                sim_list = [
                    compare.similarity_numeric(vals_internal, i) if str(i) != "" else 0
                    for i in vals_wikidata
                ]

            elif val_type == "categorical":
                sim_list = [
                    compare.similarity_categorical(
                        vals_internal, i, raise_on_diff_types=False
                    )
                    for i in vals_wikidata
                ]

            # if val_type in ['string', 'numeric', 'categorical']:
            #    print(pid, vals_internal)
            #    print(vals_wikidata)
            #    print(sim_list)

            X_temp.append(sim_list)

        X_item = np.asarray(X_temp, dtype=np.float32).transpose()
        X_list.append(X_item)

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=bool)

    return X, y
