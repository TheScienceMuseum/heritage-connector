from elasticsearch import helpers
from elasticsearch import Elasticsearch
import json

# Should we implement this as a persistance class esp. for connection pooling?
# https://elasticsearch-dsl.readthedocs.io/en/latest/persistence.html

# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es = Elasticsearch()

index = "heritageconnector"


def createIndex():
    """Delete the exiting ES index if it exists and create a new index and mappings"""

    print("Wiping existing index: " + index)
    es.indices.delete(index=index, ignore=[400, 404])

    # setup any mappings etc.
    indexSettings = {"settings": {"number_of_shards": 1, "number_of_replicas": 0}}

    print("Creating new index: " + index)
    es.indices.create(index=index, body=indexSettings)

    return


def batchCreate(data):
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
    print(es_json)

    return response


def update(collection, record_type, data, jsonld):
    """Update an existing ElasticSearch record"""

    # for now htis just mirrors the create method
    create(collection, record_type, data, jsonld)

    return


def updateGraph(id, jsonld):
    """Update the JSON-LD graph on an existing ElasticSearch record"""

    # https://www.elastic.co/guide/en/elasticsearch/reference/master/docs-update.html

    doc = {"graph": json.loads(jsonld)}
    es_json = json.dumps(doc)

    # add JSON document to ES index
    response = es.update(index=index, id=id, body=es_json)
    print(response)

    return


def delete(id):
    """Delete an existing ElasticSearch record"""

    es.delete(id)

    return


def get(id):
    """Return an existing ElasticSearch record"""

    document = es.get(index=index, id=id)

    return document


def getByURI(uri):
    """Return an existing ElasticSearch record"""

    # https://www.elastic.co/guide/en/elasticsearch/reference/master/search-search.html

    document = es.search(index=index, body={"query": {"match": {"uri": uri}}})

    return document


def search(query, filter):
    """Return an optionally filtered list of matching objects"""

    return


def sameAs(id, uri):
    """Update an existing ElasticSearch record with a new relationship"""

    # this is effectivly the same method as in the loader module
    # maybe we should move both to a HC RDF utils module?
    # def addRelationship(s, p, o):

    # "@owl:sameAs": [
    #     {
    #         "@id": "https://www.wikidata.org/wiki/1000"
    #     },
    #     {
    #         "@id": "https://www.wikidata.org/wiki/Q46633"
    #     }
    # ],

    # g = Graph().parse(data=record, format='json-ld')

    # g.add((
    #     URIRef(s),
    #     OWL.sameAs,
    #     URIRef(o),
    # ))

    # record.graph = g.serialize(format="json-ld", context=context, indent=4).decode("utf-8")

    # updateGraph(id, jsonld)

    return
