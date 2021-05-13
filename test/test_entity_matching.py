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
def fixt():
    data = pd.DataFrame.from_dict({"item_name": ["photograph", "camera", "model"]})
    rec = reconciler.Reconciler()

    map_df = rec.process_column(
        data["item_name"],
        multiple_vals=False,
        class_include="Q488383",
        class_exclude=["Q5", "Q43229", "Q28640", "Q618123", "Q16222597"],
        text_similarity_thresh=95,
    )

    return data, rec, map_df


def test_reconciler_process_column(fixt):
    data, rec, map_df = fixt
    result = reconciler.create_column_from_map_df(
        data["item_name"], map_df, multiple_vals=False
    )

    assert isinstance(result, pd.Series)
    assert result.values.tolist() == [
        ["Q125191"],
        ["Q15328"],
        ["Q1979154", "Q10929058"],
    ]


def test_reconciler_import_export(fixt):
    data, rec, map_df = fixt
    reconciler.export_map_df_to_csv(map_df, "./test_data.csv")
    imported_map_df = reconciler.import_map_df_from_csv("./test_data.csv")
    os.remove("./test_data.csv")

    # both qids and filtered_qids columns should be of type list
    assert all(isinstance(i, list) for i in imported_map_df["qids"])
    assert all(isinstance(i, list) for i in imported_map_df["filtered_qids"])


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
