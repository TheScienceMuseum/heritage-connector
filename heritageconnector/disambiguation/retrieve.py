from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.utils.wikidata import url_to_qid
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
        qids = flatten_list_of_lists(id_qid_mapping.values())

    endpoint = config.WIKIDATA_SPARQL_ENDPOINT

    sparq_qids = " ".join([f"(wd:{i})" for i in qids])
    select_slug = (
        "?"
        + " ?".join(pids_nolabel)
        + " ?"
        + " ?".join(map("{0}Label".format, pids_label))
    )
    body_exp = "\n".join([f"OPTIONAL{{ ?item wdt:{v} ?{v} .}}" for v in all_pids])

    query = f"""
        SELECT ?item ?itemLabel ?itemDescription ?altLabel {select_slug}
            WHERE {{
                VALUES (?item) {{ {sparq_qids} }}
                {body_exp}

                OPTIONAL {{
                    ?item skos:altLabel ?altLabel .
                    FILTER (lang(?altLabel) = "en")
                    }}

                SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "en" .
                }}
            }}
    """

    res = get_sparql_results(endpoint, query)["results"]["bindings"]

    res_df = pd.json_normalize(res)
    condensed = pd.DataFrame()

    if len(res_df) > 0:
        res_df = res_df[[col for col in res_df.columns if "value" in col]].rename(
            columns=lambda x: x.replace(".value", "")
        )

        # condense res_df -> multiple rows per qcode go to lists in cells
        def condense_cells(c):
            unique_list = list(set(c.dropna()))
            if len(unique_list) == 0:
                return ""
            elif len(unique_list) == 1:
                return unique_list[0]
            else:
                return unique_list

        for col in res_df.columns:
            condensed = pd.concat(
                [
                    condensed,
                    res_df.groupby("item")[col]
                    .apply(condense_cells)
                    .reset_index(drop=True),
                ],
                axis=1,
            )

        # process dataframe to return
        condensed["item"] = condensed["item"].apply(url_to_qid)

        if id_qid_mapping:
            condensed["id"] = condensed["item"].apply(
                lambda x: [key for key in id_qid_mapping if x in id_qid_mapping[key]][0]
            )
            condensed = condensed[
                ["id", "item", "itemLabel", "itemDescription", "altLabel"]
                + list(map("{0}Label".format, pids_label))
                + pids_nolabel
            ].sort_values("id")

        else:
            condensed = condensed[
                ["item", "itemLabel", "itemDescription", "altLabel"]
                + list(map("{0}Label".format, pids_label))
                + pids_nolabel
            ]

        condensed = condensed.rename(
            columns=lambda x: x.strip("Label") if x.startswith("P") else x
        )

    return condensed
