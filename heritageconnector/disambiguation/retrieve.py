from heritageconnector.utils.sparql import get_sparql_results
from heritageconnector.config import config
import pandas as pd


def get_wikidata_fields(qcodes: list, column_pid_mapping: dict) -> pd.DataFrame:
    """
    [summary]

    Args:
        qcodes (list): [description]
        column_pid_mapping (dict): [description]

    Returns:
        pd.DataFrame: [description]
    """

    endpoint = config.WIKIDATA_SPARQL_ENDPOINT

    sparq_qids = " ".join([f"(wd:{i})" for i in qcodes])

    select_slug = "?" + " ?".join(map("{0}Label".format, column_pid_mapping.keys()))
    # select_slug = "?" + " ?".join(column_pid_mapping.keys())
    body_exp = "\n".join(
        [f"OPTIONAL{{ ?item wdt:{v} ?{k} .}}" for k, v in column_pid_mapping.items()]
    )

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

        condensed = condensed[
            ["item", "itemLabel", "itemDescription", "altLabel"]
            + list(map("{0}Label".format, column_pid_mapping.keys()))
        ]
        # condensed = condensed[['item', 'itemLabel', 'itemDescription', 'altLabel'] + list(column_pid_mapping.keys())]

    return condensed
