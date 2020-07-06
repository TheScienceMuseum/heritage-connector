from elasticsearch import helpers
from elasticsearch import Elasticsearch

# Should we implement this as a persistance class esp. for connection pooling???
# https://elasticsearch-dsl.readthedocs.io/en/latest/persistence.html

# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es = Elasticsearch()

indexName = "heritageconnector"


def create_index():
    """Delete the exiting ES index if it exists and create a new index and mappings"""

    print("Wiping existing index: " + indexName)
    es.indices.delete(index=indexName, ignore=[400, 404])

    # setup any mappings etc.
    indexSettings = {"settings": {"number_of_shards": 1, "number_of_replicas": 0}}

    print("Creating new index: " + indexName)
    es.indices.create(index=indexName, body=indexSettings)

    return


def batch_create(data):
    """Batch load a set of new records into ElasticSearch"""

    # todo
    # https://elasticsearch-py.readthedocs.io/en/master/helpers.html#helpers

    return


def create(data, jsonld):
    """Load a new record in ElasticSearch and return it's id"""

    # create a ES doc
    doc = {}
    doc["graph"] = jsonld

    # add document to ES index
    # todo: should we use out own ID/key? Much nicer but what to use as a uniqye value? url?
    response = es.index(index=indexName, body=doc)
    print(response)

    return id


def get(id):
    """Return an existing ElasticSearch record"""

    res = es.get(index=indexName, id=id)
    print(res["_source"])

    data = res
    return data


def update(id):
    """Overwrite an existing ElasticSearch record"""

    # todo
    # should we also have a 'utility' method that adds a 'triple' to an exiting ES record?

    return


def delete(id):
    """Overwrite an existing ElasticSearch record"""

    # delete

    return
