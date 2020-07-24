from heritageconnector import datastore
from rdflib import Graph, Literal, RDF, URIRef, BNode
from rdflib.namespace import XSD, FOAF, OWL
from rdflib.serializer import Serializer
from rdflib.plugins.stores import sparqlstore
from logging import getLogger
import json

# Fuseki endpoints (localhost)
query_endpoint = "http://localhost:3030/ds/query"
update_endpoint = "http://localhost:3030/ds/update"


def my_bnode_ext(node):
    if isinstance(node, BNode):
        return "<bnode:b%s>" % node
    return sparqlstore._node_to_sparql(node)


def open_sparql_store():
    store = sparqlstore.SPARQLUpdateStore(node_to_sparql=my_bnode_ext)
    store.open((query_endpoint, update_endpoint))
    return store


def export_to_sparql_store():

    store = open_sparql_store()
    g = Graph(store, identifier="urn:x-arq:DefaultGraph")

    # Add object records / triples
    records = datastore.get_by_type("object")
    for record in records:
        jsonld = json.dumps(record["_source"]["graph"])
        g.parse(data=jsonld, format="json-ld")

    # Add people records / triples
    records = datastore.get_by_type("person")
    for record in records:
        jsonld = json.dumps(record["_source"]["graph"])
        g.parse(data=jsonld, format="json-ld")

    return


if __name__ == "__main__":

    export_to_sparql_store()