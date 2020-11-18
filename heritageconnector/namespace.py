from rdflib import Namespace, URIRef, Literal
from rdflib.namespace import XSD, FOAF, OWL, RDF, RDFS, PROV, SDO, SKOS
from typing import Union, List

# --- STORE FOR ALL NAMESPACES USED IN THE HERITAGE CONNECTOR ---
# --- to import, use `from heritageconnector.namespace import *` ---

WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

# --- Custom Namespaces
SMGP = Namespace("https://collection.sciencemuseumgroup.org.uk/people/")
SMGO = Namespace("https://collection.sciencemuseumgroup.org.uk/objects/")
SMGD = Namespace("https://collection.sciencemuseumgroup.org.uk/documents/")

# --- Tuple of internal namespaces
_internal = SMGP, SMGO, SMGD


def is_internal_uri(val: Union[str, URIRef, Literal]) -> bool:
    """
    Returns whether a given value (string/rdflib.URIRef/rdflib.Literal) is an internal URI according to heritageconnector.namespace._internal.
    """

    return any([str(val).startswith(str(namespace)) for namespace in _internal])
