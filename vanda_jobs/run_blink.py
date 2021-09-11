import sys

sys.path.append("..")

from heritageconnector.nlp.nel import BLINKServiceWrapper
from heritageconnector.config import config
from heritageconnector.utils.generic import get_timestamp

if __name__ == "__main__":
    base_url = sys.argv[1]
    endpoint = "http://" + base_url + ":8000/blink/multiple"

    entity_fields = [
        "graph.@hc:entityPERSON.@value",
        "graph.@hc:entityORG.@value",
        "graph.@hc:entityLOC.@value",
        "graph.@hc:entityFAC.@value",
        "graph.@hc:entityOBJECT.@value",
        "graph.@hc:entityLANGUAGE.@value",
        "graph.@hc:entityNORP.@value",
        "graph.@hc:entityEVENT.@value",
        # "graph.@hc:entityDATE.@value",
    ]

    threshold = 0.8

    blink_service = BLINKServiceWrapper(
        endpoint,
        description_field="data.http://www.w3.org/2001/XMLSchema#description",
        entity_fields=entity_fields,
        wiki_link_threshold=threshold,
    )

    blink_service.process_unlinked_entity_mentions(
        config.ELASTIC_SEARCH_INDEX,
        f"../GITIGNORE_DATA/vanda_hc_data/NEL/blink_output_{get_timestamp()}.jsonl",
        page_size=12,
        limit=None,
    )
