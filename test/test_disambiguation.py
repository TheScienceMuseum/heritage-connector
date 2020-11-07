from heritageconnector.disambiguation import retrieve, search
import re


def test_get_wikidata_fields():
    qids = ["Q203545", "Q706475", "Q18637243"]  # all humans
    pids = ["P31", "P21", "P735", "P734", "P1971", "P36"]
    pids_nolabel = ["P21", "P31"]

    res_df = retrieve.get_wikidata_fields(
        pids=pids, qids=qids, pids_nolabel=pids_nolabel
    )

    assert res_df.shape == (3, 14)
    assert set(res_df.columns.tolist()) == set(
        ["qid", "label", "description", "aliases"]
        + pids
        + [pid + "Label" for pid in pids if pid not in pids_nolabel]
    )

    # all values in nolabels cols should be QIDs or empty
    vals_nolabel = res_df["P31"].tolist() + res_df["P21"].tolist()
    assert all(
        [(len(re.findall(r"(Q\d+)", val)) == 1) or val == "" for val in vals_nolabel]
    )

    # no humans have property P36 (capital) so all values should be empty strings
    assert all([val == "" for val in res_df["P36"].tolist()])


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
    assert len(list(set(res1))) == len(res1)
    assert len(list(set(res2))) == len(res2)

    # this result will contain 'phonograph' as well without the `return_exact_only` flag
    res3 = s.run_search(
        "photograph", return_exact_only=True, field_exists_filter="claims.P279",
    )

    assert res3 == ["Q125191"]
