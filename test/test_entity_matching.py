import sys

sys.path.append("..")

# ignore pandas futurewarning
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import pytest
import pandas as pd
from heritageconnector.entity_matching import reconciler, lookup


@pytest.fixture
def rec():
    data = pd.DataFrame.from_dict({"item_name": ["photograph", "camera", "model"]})
    rec = reconciler.Reconciler(data, table="OBJECT")

    rec.process_column(
        "item_name",
        multiple_vals=False,
        class_include="Q223557",
        class_exclude=["Q5", "Q43229", "Q28640", "Q618123"],
        text_similarity_thresh=95,
    )

    return rec


def test_reconciler_process_column(rec):
    with pytest.warns(None) as record:
        result = rec.create_column_from_map_df("item_name")

    assert len(record) == 1
    assert isinstance(result, pd.Series)
    assert result.values.tolist() == [["Q125191"], ["Q15328"], ["Q57312861"]]


def test_reconciler_import_export(rec):
    rec.export_map_df("./test_data.csv")
    rec.import_map_df("./test_data.csv")
    os.remove(rec.current_file_path)

    # both qids and filtered_qids columns should be of type list
    assert all(isinstance(i, list) for i in rec._map_df_imported["qids"])
    assert all(isinstance(i, list) for i in rec._map_df_imported["filtered_qids"])


def test_get_sameas_links_from_external_id():
    res = lookup.get_sameas_links_from_external_id("P4389")

    assert isinstance(res, pd.DataFrame)
    assert res.columns.values.tolist() == ["wikidata_url", "external_url"]
    assert all(
        [
            i.startswith("http://www.wikidata.org/entity/")
            for i in res["wikidata_url"].tolist()
        ]
    )
