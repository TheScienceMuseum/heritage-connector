from elasticsearch import helpers
from elasticsearch import Elasticsearch
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import XSD, FOAF, OWL
from rdflib.serializer import Serializer
import json

# Should we implement this as a persistance class esp. for connection pooling?
# https://elasticsearch-dsl.readthedocs.io/en/latest/persistence.html

# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
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

        # re-serialise the graoh and update the reccord
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


def get(id):
    """Return an existing ElasticSearch record"""

    document = es.get(index=index, id=id)

    return document


def get_by_uri(uri):
    """Return an existing ElasticSearch record"""

    res = es.search(index=index, body={"query": {"match": {"uri": uri}}})
    if len(res["hits"]["hits"]):
        return res["hits"]["hits"][0]
    else:
        return


def get_by_type(type):
    """Return an list of matching ElasticSearch record"""

    # object
    # person
    # organisation
    # document
    # artcile

    return


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
    # datastore.update_graph(URIRef(maker_uri), FOAF.made, URIRef(uri))

    return response


def add_user(uri, relationship, user_uri):
    """Adds a user relationship to an existing record"""

    # TODO: need to find a RDF term foor USER/USED?
    response = update_graph(uri, FOAF.maker, user_uri)
    # datastore.update_graph(URIRef(user_uri), FOAF.made, URIRef(uri))

    return response
