import json
from typing import List


def get_entities_from_jsonl_doc(
    doc: dict,
    ent_types=[
        "PERSON",
        "ORG",
        "FAC",
        "LOC",
        "OBJECT",
    ],
) -> dict:
    """Get dict of {uri: _, description: _, entities: [(text, TYPE), ...]} from a JSON-LD doc"""
    _id = doc["_id"]
    description = doc["_source"]["data"]["http://www.w3.org/2001/XMLSchema#description"]

    doc_entity_vals = {
        k: v for k, v in doc["_source"]["graph"].items() if k.startswith("@hc:entity")
    }

    doc_entity_vals_simplified = {
        "uri": _id,
        "description": description,
        "entities": [],
    }

    for k, v in doc_entity_vals.items():
        if isinstance(v, dict):
            simplified_v = [v["@value"]]
        elif isinstance(v, list):
            simplified_v = [item["@value"] for item in v]

        simplified_k = k[10:]

        if simplified_k in ent_types:
            doc_entity_vals_simplified["entities"] += [
                (mention, simplified_k) for mention in simplified_v
            ]
    #             doc_entity_vals_simplified["entities"].update({simplified_k: simplified_v})

    return doc_entity_vals_simplified


def load_jsonld_dump(data_path: str) -> List[dict]:
    with open(data_path, "r") as json_file:
        json_list = [json.loads(item) for item in list(json_file)]

    return json_list
