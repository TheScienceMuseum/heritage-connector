from elasticsearch import helpers
from elasticsearch import Elasticsearch
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.serializer import Serializer
import json
from tqdm.auto import tqdm
from itertools import islice
import os
from heritageconnector.namespace import XSD, FOAF, OWL, PROV
from heritageconnector.config import config
from heritageconnector import logging, errors

logger = logging.get_logger(__name__)

# Should we implement this as a persistance class esp. for connection pooling?
# https://elasticsearch-dsl.readthedocs.io/en/latest/persistence.html

if hasattr(config, "ELASTIC_SEARCH_CLUSTER"):
    es = Elasticsearch(
        [config.ELASTIC_SEARCH_CLUSTER],
        http_auth=(config.ELASTIC_SEARCH_USER, config.ELASTIC_SEARCH_PASSWORD),
    )
else:
    # use localhost
    es = Elasticsearch()

index = config.ELASTIC_SEARCH_INDEX
es_config = {
    "chunk_size": int(config.ES_BULK_CHUNK_SIZE),
    "queue_size": int(config.ES_BULK_QUEUE_SIZE),
}

context = [
    {"@foaf": "http://xmlns.com/foaf/0.1/", "@language": "en"},
    {"@schema": "http://www.w3.org/2001/XMLSchema#", "@language": "en"},
    {"@owl": "http://www.w3.org/2002/07/owl#", "@language": "en"},
]


def create_index():
    """Delete the exiting ES index if it exists and create a new index and mappings"""

    logger.info("Wiping existing index: " + index)
    es.indices.delete(index=index, ignore=[400, 404])

    # setup any mappings etc.
    indexSettings = {"settings": {"number_of_shards": 1, "number_of_replicas": 0}}

    logger.info("Creating new index: " + index)
    es.indices.create(index=index, body=indexSettings)


def es_bulk(action_generator, total_iterations=None):
    """Batch load a set of new records into ElasticSearch"""

    successes = 0
    errs = []

    for ok, action in tqdm(
        helpers.parallel_bulk(
            client=es,
            index=index,
            actions=action_generator,
            chunk_size=es_config["chunk_size"],
            queue_size=es_config["queue_size"],
            raise_on_error=False,
        ),
        total=total_iterations,
    ):
        if not ok:
            errs.append(action)
        successes += ok

    return successes, errs


def create(collection, record_type, data, jsonld):
    """Load a new record in ElasticSearch and return its id"""

    # create a ES doc
    doc = {
        "uri": data["uri"],
        "collection": collection,
        "type": record_type,
        "data": {i: data[i] for i in data if i != "uri"},
        "graph": json.loads(jsonld),
    }
    es_json = json.dumps(doc)

    # add JSON document to ES index
    response = es.index(index=index, id=data["uri"], body=es_json)

    return response


def update_graph(s_uri, p, o_uri):
    """Add a new RDF relationship to an an existing record"""

    # create graph containing just the new triple
    g = Graph()
    g.add((URIRef(s_uri), p, URIRef(o_uri)))

    # export triple as JSON-LD and remove ID, context
    jsonld_dict = json.loads(g.serialize(format="json-ld", context=context, indent=4))
    _ = jsonld_dict.pop("@id")
    _ = jsonld_dict.pop("@context")

    body = {"doc": {"graph": jsonld_dict}}

    es.update(index=index, id=s_uri, body=body, ignore=404)


def delete(id):
    """Delete an existing ElasticSearch record"""

    es.delete(id)


def get_by_uri(uri):
    """Return an existing ElasticSearch record"""

    res = es.search(index=index, body={"query": {"term": {"uri.keyword": uri}}})
    if len(res["hits"]["hits"]):
        return res["hits"]["hits"][0]


def get_by_type(type, size=1000):
    """Return an list of matching ElasticSearch record"""

    res = es.search(index=index, body={"query": {"match": {"type": type}}}, size=size)
    return res["hits"]["hits"]


def get_graph(uri):
    """Return an the RDF graph for an ElasticSearch record"""

    record = get_by_uri(uri)
    if record:
        jsonld = json.dumps(record["_source"]["graph"])
        g = Graph().parse(data=jsonld, format="json-ld")

    return g


def get_graph_by_type(type):
    """Return an list of matching ElasticSearch record"""

    g = Graph()
    records = get_by_type(type)
    for record in records:
        jsonld = json.dumps(record["_source"]["graph"])
        g.parse(data=jsonld, format="json-ld")

    return g


def add_same_as(s_uri, o_uri):
    """Adds a sameAs relationship to an existing record"""

    update_graph(s_uri, OWL.sameAs, o_uri)


def add_maker(uri, relationship, maker_uri):
    """Adds a maker relationship to an existing record"""

    update_graph(uri, FOAF.maker, maker_uri)


def add_user(uri, relationship, user_uri):
    """Adds a user relationship to an existing record"""

    update_graph(user_uri, PROV.used, uri)


def es_to_rdflib_graph(g=None, return_format=None):
    """
    Turns a dump of ES index into an RDF format. Returns an RDFlib graph object if no
    format is specified, else an object with the specified format which could be written
    to a file.
    """

    # get dump
    res = helpers.scan(
        client=es, index=index, query={"_source": "graph.*", "query": {"match_all": {}}}
    )
    total = es.count(index=index)["count"]

    # create graph
    if g is None:
        g = Graph()

        for item in tqdm(res, total=total):
            g += Graph().parse(
                data=json.dumps(item["_source"]["graph"]), format="json-ld"
            )
    else:
        logger.debug("Using existing graph")
        for item in tqdm(res, total=total):
            g.parse(data=json.dumps(item["_source"]["graph"]), format="json-ld")

    if return_format is None:
        return g
    else:
        return g.serialize(format=return_format)


def es_text_to_json(json_path: str, limit: int = None):
    """
    Saves an Elasticsearch dump to a json file in the location specified by `json_path`.
    JSON file has keys 'uri', 'text'.
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
