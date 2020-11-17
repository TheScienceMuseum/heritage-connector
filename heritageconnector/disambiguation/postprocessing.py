"""
Tools to process the results from disambiguation.pipelines.
All methods here expect a dataframe `pairs` with columns internal_id, wikidata_id, is_type, y_pred, y_pred_proba, and return a filtered
dataframe with the same columns.
"""

import pandas as pd
from typing import Union

from heritageconnector.disambiguation.retrieve import get_wikidata_fields
from heritageconnector.utils.wikidata import wbentities

entities = wbentities()


def _check_pairs_columns(pairs: pd.DataFrame):
    if set(pairs.columns.tolist()) != {
        "internal_id",
        "wikidata_id",
        "y_pred",
        "y_pred_proba",
        "is_type",
    }:
        raise ValueError(
            "Input dataframe does not contain correct columns (internal_id, wikidata_id, y_pred, y_pred_proba, is_type)."
            "Make sure it has been created by disambiguator.predict_top_ranked_pairs."
        )


def enforce_correct_type(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the rows of pairs where the type of the internal record and Wikidata record match.
    """

    _check_pairs_columns(pairs)

    pairs_new = pairs.copy()

    return pairs_new[pairs_new["is_type"] == True]  # noqa: E712


def filter_max_wikidata_links(
    pairs: pd.DataFrame, max_wikidata_links: int = 3
) -> pd.DataFrame:
    """
    Return only internal records that have less than or equal to `max_links` predicted links to Wikidata items.
    """

    _check_pairs_columns(pairs)

    pairs_new = pairs.copy()
    links_per_internal_id = pairs_new["internal_id"].value_counts()
    filtered_internal_ids = links_per_internal_id[
        links_per_internal_id <= max_wikidata_links
    ].index.tolist()
    pairs_new = pairs_new[pairs_new["internal_id"].isin(filtered_internal_ids)]

    return pairs_new


def filter_cased_wikidata_labels(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Return only record pairs where the label of the Wikidata record contains a capital letter (i.e. removes
    items whose Wikidata labels are all lowercase). Useful for filtering out generic records which tend to
    refer to types of things rather than things themselves, e.g. 'electric car' (Q193692) or 'synthesizer'
    (Q163829).

    Only needs column `wikidata_id`.
    """

    if "wikidata_id" not in pairs.columns.values:
        raise ValueError("DataFrame pairs must contain column `wikidata_id`.")

    qid_label_df = get_wikidata_fields(pids=[], qids=pairs["wikidata_id"].tolist()).loc[
        :, ["qid", "label"]
    ]

    qids_with_non_lowercase_label = qid_label_df.loc[
        qid_label_df["label"] != qid_label_df["label"].str.lower(), "qid"
    ].tolist()

    pairs_new = pairs.copy()

    return pairs_new[pairs_new["wikidata_id"].isin(qids_with_non_lowercase_label)]


def remove_wikidata_items_with_no_claims(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Return only record pairs with a non-zero number of claims. Only needs column `wikidata_id`.
    """

    if "wikidata_id" not in pairs.columns.values:
        raise ValueError("DataFrame pairs must contain column `wikidata_id`.")

    qid_claim_counts = entities.get_count_of_claims(pairs["wikidata_id"].tolist())
    qids_with_claims = [qid for qid, count in qid_claim_counts.items() if count > 0]

    pairs_new = pairs.copy()

    return pairs_new[pairs_new["wikidata_id"].isin(qids_with_claims)]
