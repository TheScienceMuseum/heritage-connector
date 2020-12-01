from elasticsearch import Elasticsearch, helpers
from tqdm.auto import tqdm
import json
from itertools import islice
import pandas as pd
from heritageconnector.config import config
from heritageconnector.datastore import es, index
from heritageconnector import logging

logger = logging.get_logger(__name__)

topconcept_to_spacy_ner_mapping = {
    "PERSON": "PERSON",
    "ORGANISATION": "ORG",
    "OBJECT": "PRODUCT",
}


def descriptions_to_json(json_path: str, limit: int = None):
    """
    Saves an Elasticsearch dump to a json file in the location specified by `json_path`.
    JSON file has keys 'uri', 'text'.
    Output example:
    ``` json
    [{"uri": "http://a.URI","text":"Two-minute phonograph cylinder"},
     {"uri": "http://another.URI, "text": "'Rover' Safety Bicycle"}]
    ```
    """

    limit = -1 if limit is None else limit
    text_field = "data.http://www.w3.org/2001/XMLSchema#description"

    res = helpers.scan(
        client=es,
        index=index,
        query={"_source": f"{text_field}", "query": {"match_all": {}}},
        preserve_order=True,
    )

    total = es.count(index=index)["count"]

    items_out = []
    item_count = 0

    logger.info(f"Getting documents with field {text_field}")
    for item in tqdm(res, total=total):
        # if there is no 'graph' key then `text_field` does not appear in the document
        if "data" in item["_source"]:
            items_out.append(
                {
                    "uri": item["_id"],
                    "text": item["_source"]["data"][
                        "http://www.w3.org/2001/XMLSchema#description"
                    ],
                }
            )
            item_count += 1

        if item_count == limit:
            break

    logger.info(f"{len(items_out)} items exporting to {json_path}")
    with open(json_path, "w") as f:
        json.dump(items_out, f)


def labels_ids_to_jsonl(
    jsonl_path: str,
    include_aliases: bool = True,
    include_ids: bool = True,
    limit: int = None,
    drop_duplicates_across_types: bool = True,
    topconcepts_to_ignore: list = None,
):
    """
    Export labels and IDs to a JSONL file

    Args:
        jsonl_path (str): path to JSONL file
        include_aliases (bool, optional): Whether to include aliases as well as labels. Defaults to True.
        include_ids (bool, optional): Whether to include IDs, which could be used to link the labels back to their associated records.
        limit (int, optional): Only extract the first `limit` names.
        drop_duplicates_across_types (bool, optional): Whether to drop duplicates if they appear across more than one type. E.g. "Salvador Dali"
            is both a person and an object (PRODUCT) in the SMG collection.
    """

    res = helpers.scan(
        client=es,
        index=index,
        query={"_source": "graph", "query": {"match_all": {}}},
        preserve_order=True,
    )

    if limit is None:
        limit = es.count(index=index)["count"]
    else:
        res = islice(res, limit)

    items_out = []

    for item in tqdm(res, total=limit):
        graph = item["_source"]["graph"]

        item_id = graph["@id"]

        try:
            item_topconcept = graph["@skos:hasTopConcept"]["@value"]

            if item_topconcept.upper() in topconcepts_to_ignore:
                continue

            if item_topconcept == "PERSON":
                item_label = (
                    str(graph["@foaf:givenName"]["@value"])
                    + " "
                    + str(graph["@foaf:familyName"]["@value"])
                )

            else:
                item_label = graph["@rdfs:label"]["@value"]

            items_out.append(
                {
                    "label": topconcept_to_spacy_ner_mapping[item_topconcept],
                    "pattern": item_label,
                    "id": item_id,
                }
            )

        except:  # noqa: E722
            # logger.debug(f"{item_id} skipped")
            pass

    if drop_duplicates_across_types:
        df = pd.DataFrame(items_out)
        duplicated_labels = (
            df["pattern"]
            .value_counts()[df["pattern"].value_counts() > 1]
            .index.tolist()
        )
        no_unique_types_per_duplicated_label = (
            df.loc[df["pattern"].isin(duplicated_labels)]
            .groupby("pattern")["label"]
            .nunique()
        )
        labels_with_multiple_types = (
            no_unique_types_per_duplicated_label[
                no_unique_types_per_duplicated_label > 1
            ]
            .index.str.lower()
            .tolist()
        )

        items_out = df[
            ~df["pattern"].str.lower().isin(labels_with_multiple_types)
        ].to_dict("records")

    with open(jsonl_path, "w") as f:
        for item in items_out:
            json.dump(item, f)
            f.write("\n")
