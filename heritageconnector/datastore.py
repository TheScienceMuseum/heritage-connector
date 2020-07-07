from elasticsearch import helpers
from elasticsearch import Elasticsearch

# Should we implement this as a persistance class esp. for connection pooling???
# https://elasticsearch-dsl.readthedocs.io/en/latest/persistence.html

# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es = Elasticsearch()

indexBase = "heritageconnector"


def createIndex():
    """Delete the exiting ES index if it exists and create a new index and mappings"""

    print("Wiping existing index: " + indexBase)
    es.indices.delete(index=indexBase, ignore=[400, 404])

    # setup any mappings etc.
    indexSettings = {"settings": {"number_of_shards": 1, "number_of_replicas": 0}}

    print("Creating new index: " + indexBase)
    es.indices.create(index=indexBase, body=indexSettings)

    return


def batchCreate(data):
    """Batch load a set of new records into ElasticSearch"""

    # todo
    # https://elasticsearch-py.readthedocs.io/en/master/helpers.html#helpers

    return


def create(collection, record_type, data, jsonld):
    """Load a new record in ElasticSearch and return it's id"""

    # create a ES doc
    doc = {}
    doc["graph"] = jsonld

    # add document to ES index
    # todo: should we use out own ID/key? Much nicer but what to use as a uniqye value? url?
    response = es.index(index=indexBase, body=doc)
    print(response)

    return id


def delete(id):
    """Overwrite an existing ElasticSearch record"""

    # delete

    return


def getAll(id):
    """Return an list of objects or make this an itterator"""

    return


def get(id):
    """Return an existing ElasticSearch record"""

    res = es.get(index=indexBase, id=id)
    print(res["_source"])

    data = res
    return data


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

    return
