from heritageconnector.disambiguation import retrieve, search


def test_get_wikidata_fields():
    res_df = retrieve.get_wikidata_fields(
        pids=["P569", "P570"], qids=["Q106481", "Q46633"]
    )

    assert res_df.shape == (2, 6)
    assert set(res_df.columns.tolist()) == set(
        ["item", "itemLabel", "itemDescription", "altLabel", "P569", "P570"]
    )


def test_es_text_search():
    s = search.es_text_search()

    # this unconstrained search should return many results
    res1 = s.run_search(
        "museum", return_instanceof=False, limit=500, similarity_thresh=95
    )

    # this constrained search should return fewer results, assuming the Wikidata import has not already been filtered by P279
    res2 = s.run_search(
        "museum",
        return_instanceof=False,
        limit=5000,
        similarity_thresh=95,
        field_exists_filter="claims.P279",
    )

    assert len(res1) > 0
    assert len(res2) > 0
    assert len(res2) < len(res1)
