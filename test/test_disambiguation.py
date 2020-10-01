from heritageconnector.disambiguation import retrieve


def test_get_wikidata_fields():
    res_df = retrieve.get_wikidata_fields(
        pids=["P569", "P570"], qids=["Q106481", "Q46633"]
    )

    assert res_df.shape == (2, 6)
    assert set(res_df.columns.tolist()) == set(
        ["item", "itemLabel", "itemDescription", "altLabel", "P569", "P570"]
    )
