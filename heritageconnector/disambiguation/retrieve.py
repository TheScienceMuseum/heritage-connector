from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.utils.wikidata import (
    wbentities,
    url_to_qid,
    join_qids_for_sparql_values_clause,
)
from heritageconnector.utils.generic import flatten_list_of_lists
from heritageconnector.config import config
import pandas as pd


def get_wikidata_fields(
    pids: list, qids: list = [], id_qid_mapping: dict = {}, pids_nolabel: list = []
) -> pd.DataFrame:
    """
    Get information for Wikidata items specified by a set of Wikidata QIDs. Return columns specified by a set of Wikidata PIDs.
        Optionally provide an internal ID-QID mapping to get the results grouped by internal ID.

    Args:
        pids (list): list of Wikidata PIDs
        qcodes (list, optional): list of Wikidata QIDs
        id_qcode_mapping (dict, optional): {internal_id: [qids], ...}. ID column is added to returned DataFrame to retain this 
            mapping
        pids_nolabel (list, optional): PIDs for which the value should be returned instead of the label. Any pids not included 
            in `pids` will be added to the final result.

    Returns:
        pd.DataFrame: table of Wikidata results
    """

    all_pids = list(set(pids + pids_nolabel))
    pids_label = list(set(all_pids) - set(pids_nolabel))

    if qids and id_qid_mapping:
        raise ValueError("Only one of qids and id_qid_mapping should be provided.")
    elif id_qid_mapping:
        qids = list(set(flatten_list_of_lists(id_qid_mapping.values())))

    ent = wbentities()
    ent.get_properties(qids, all_pids, pids_label)
    res_df = (
        ent.get_results()
        .rename(columns={"id": "qid", "labels": "label", "descriptions": "description"})
        .rename(columns=lambda c: c.strip("claims.") if c.startswith("claims.") else c)
    )

    # this line checks that all the QIDs that were requested have ended up in the resulting dataframe
    assert len(set(qids)) == len(set(res_df["qid"].tolist()))

    if id_qid_mapping:
        res_df["id"] = res_df["qid"].apply(
            lambda x: [key for key in id_qid_mapping if x in id_qid_mapping[key]][0]
        )
        res_df = res_df.sort_values("id")

    return res_df
