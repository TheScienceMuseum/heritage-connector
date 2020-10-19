from heritageconnector.disambiguation import retrieve, pipelines
from heritageconnector.namespace import OWL, SKOS
import numpy as np
from rdflib import URIRef, Literal
import pytest


def test_disambiguator_process_wbgetentities_results():
    qids = ["Q2897681", "Q75931117", "Q6198902", "Q3805088"]
    pids = ["P735", "P734", "P21", "P569", "P570", "P106", "P31"]

    d = pipelines.Disambiguator()
    wikidata_results = retrieve.get_wikidata_fields(pids, qids)

    results_processed = d._process_wikidata_results(wikidata_results)
    dates_processed = (
        results_processed["P569"].tolist() + results_processed["P570"].tolist()
    )

    # years can be converted to int
    assert all([(str(int(val)) == str(val)) for val in dates_processed if val != ""])


@pytest.mark.skip(reason="relies on local fuseki instance running")
def test_disambiguator_get_unique_predicates():
    predicates_ignore = [OWL.sameAs, SKOS.hasTopConcept]

    d = pipelines.Disambiguator()
    res = d._get_predicates_for_top_concept(
        "PERSON", predicates_ignore=predicates_ignore
    )

    assert len(res) > 0
    assert len(set(res)) == len(res)  # unique values
    assert (
        len(set(res).intersection(set(predicates_ignore))) == 0
    )  # none of the predicates to ignore in the list


@pytest.mark.skip(reason="relies on local fuseki instance running")
def test_disambiguator_get_triples():
    d = pipelines.Disambiguator()
    res = d._get_triples_from_store(
        (
            None,
            URIRef("http://www.w3.org/2004/02/skos/core#hasTopConcept"),
            Literal("PERSON"),
        )
    )
    res_list = [i for i in res]

    assert len(res_list) > 0


def test_disambiguator_make_training_data():
    d = pipelines.Disambiguator()
    X, y, X_columns, id_pair_list = d.build_training_data(
        True, "PERSON", page_size=100, limit=200, search_limit=10
    )

    # array sizes
    assert X.shape[0] == y.shape[0] == len(id_pair_list)
    assert X.shape[1] == len(X_columns)

    # values
    assert (X >= 0).all()
    assert (X <= 1).all()

    # for different types
    for idx, col in enumerate(X_columns):
        if col in ["label", "P735", "P734", "P31", "P569", "P570"]:
            # text and numerical similarity are continuous, so some values won't
            # exactly round to 2 decimal places
            assert (X[:, idx].round(2) != X[:, idx]).any()

        elif col in ["P21", "P106"]:
            # categorical similarity is in [0,1]
            assert (np.isin(X[:, idx], [0, 1])).all()
