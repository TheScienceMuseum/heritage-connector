import sys

sys.path.append("..")

# ignore pandas futurewarning
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
from heritageconnector.entity_matching import reconciler


def test_reconciler_process_column():
    data = pd.DataFrame.from_dict({"item_name": ["photograph", "camera", "model"]})

    rec = reconciler.reconciler(data, table="OBJECT")
    result = rec.process_column(
        "item_name",
        multiple_vals=False,
        class_include="Q223557",
        class_exclude=["Q5", "Q43229", "Q28640", "Q618123"],
        text_similarity_thresh=95,
    )

    assert isinstance(result, pd.Series)
    assert result.values.tolist() == [["Q125191"], ["Q15328"], ["Q57312861"]]
