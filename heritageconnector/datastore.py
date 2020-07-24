from elasticsearch import helpers
from elasticsearch import Elasticsearch
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import XSD, FOAF, OWL
from rdflib.serializer import Serializer
from heritageconnector.config import config
import json

# Should we implement this as a persistance class esp. for connection pooling?
# https://elasticsearch-dsl.readthedocs.io/en/latest/persistence.html

if hasattr(config, "ELASTIC_SEARCH_CLUSTER"):
    es = Elasticsearch(
        cloud_id=config.ELASTIC_SEARCH_CLUSTER,
        http_auth=(config.ELASTIC_SEARCH_USER, config.ELASTIC_SEARCH_PASSWORD),
    )
else:
    es = Elasticsearch()

index = "heritageconnector"

context = [
    {"@foaf": "http://xmlns.com/foaf/0.1/", "@language": "en"},
    {"@schema": "http://www.w3.org/2001/XMLSchema#", "@language": "en"},
    {"@owl": "http://www.w3.org/2002/07/owl#", "@language": "en"},
]


def create_index():
    """Delete the exiting ES index if it exists and create a new index and mappings"""

    print("Wiping existing index: " + index)
    es.indices.delete(index=index, ignore=[400, 404])

    # setup any mappings etc.
    indexSettings = {"settings": {"number_of_shards": 1, "number_of_replicas": 0}}

    print("Creating new index: " + index)
    es.indices.create(index=index, body=indexSettings)

    return


def batch_create(data):
    """Batch load a set of new records into ElasticSearch"""

    # todo
    # https://elasticsearch-py.readthedocs.io/en/master/helpers.html#helpers

    return


def create(collection, record_type, data, jsonld):
    """Load a new record in ElasticSearch and return it's id"""

    # should we make our own ID using the subject URI?

    # create a ES doc
    doc = {
        "uri": data["uri"],
        "collection": collection,
        "type": record_type,
        "graph": json.loads(jsonld),
    }
    es_json = json.dumps(doc)

    # add JSON document to ES index
    response = es.index(index=index, body=es_json)

    print("Created ES record " + data["uri"])

    return response


def update():
    """Update an existing ElasticSearch record"""

    return


def update_graph(s_uri, p, o_uri):
    """Add a new RDF relationship to an an existing record"""

    # Can we do this more efficently ie. just add the new tripple to the graph and add the updates in batches    # Do we do the lookup against out config file here? (I think yes)
    # Do we store multiple entries for both Wikidata and RDF? (I think yes)

    record = get_by_uri(s_uri)
    if record:
        jsonld = json.dumps(record["_source"]["graph"])
        uid = record["_id"]
        g = Graph().parse(data=jsonld, format="json-ld")

        # add the new tripple / RDF statement to the existing graph
        g.add((URIRef(s_uri), p, URIRef(o_uri)))

        # re-serialise the graph and update the reccord
        jsonld = g.serialize(format="json-ld", context=context, indent=4).decode(
            "utf-8"
        )

        # create a ES doc
        doc = {
            "uri": record["_source"]["uri"],
            "collection": record["_source"]["collection"],
            "type": record["_source"]["type"],
            "graph": json.loads(jsonld),
        }
        es_json = json.dumps(doc)

        # Overwrite existing ES record
        response = es.index(index=index, id=uid, body=es_json)

        print("Updated ES record" + uid + " : " + record["_source"]["uri"])

    return response


def delete(id):
    """Delete an existing ElasticSearch record"""

    es.delete(id)

    return


def get_by_uri(uri):
    """Return an existing ElasticSearch record"""

    res = es.search(index=index, body={"query": {"match": {"uri": uri}}})
    if len(res["hits"]["hits"]):
        return res["hits"]["hits"][0]
    else:
        return


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


def search(query, filter):
    """Return an optionally filtered list of matching objects"""

    return


def add_same_as(s_uri, o_uri):
    """Adds a sameAs relationship to an existing record"""

    response = update_graph(s_uri, OWL.sameAs, o_uri)

    return response


def add_maker(uri, relationship, maker_uri):
    """Adds a maker relationship to an existing record"""

    response = update_graph(uri, FOAF.maker, maker_uri)
    # update_graph(URIRef(maker_uri), FOAF.made, URIRef(uri))

    return response


def add_user(uri, relationship, user_uri):
    """Adds a user relationship to an existing record"""

    # TODO: need to find a RDF term foor USER/USED?
    response = update_graph(uri, FOAF.knows, user_uri)
    # update_graph(URIRef(user_uri), FOAF.made, URIRef(uri))

    return response


def es_to_rdflib_graph(return_format=None):
    """
    Turns a dump of ES index into an RDF format. Returns an RDFlib graph object if no
    format is specified, else an object with the specified format which could be written
    to a file.
    """

    # get dump
    res = helpers.scan(
        client=es, index=index, query={"_source": "graph.*", "query": {"match_all": {}}}
    )

    # hits = res["hits"]["hits"]

    # create graph
    g = Graph()
    for item in res:
        g += Graph().parse(data=json.dumps(item["_source"]["graph"]), format="json-ld")

    if return_format is None:
        return g
    else:
        return g.serialize(format=return_format)
