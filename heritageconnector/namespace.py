from rdflib import Namespace, URIRef, Literal
from rdflib.namespace import ClosedNamespace, XSD, FOAF, OWL, RDF, RDFS, PROV, SDO, SKOS
from typing import Union, List

# --- STORE FOR ALL NAMESPACES USED IN THE HERITAGE CONNECTOR ---
# --- to import, use `from heritageconnector.namespace import *` ---

WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

# --- Custom Namespaces
SMGP = Namespace("https://collection.sciencemuseumgroup.org.uk/people/")
SMGO = Namespace("https://collection.sciencemuseumgroup.org.uk/objects/")
SMGD = Namespace("https://collection.sciencemuseumgroup.org.uk/documents/")
HC = ClosedNamespace(
    uri=URIRef("http://www.heritageconnector.org/RDF/"),
    terms=[
        "entityPERSON",
        "entityORG",
        "entityNORP",
        "entityFAC",
        "entityLOC",
        "entityOBJECT",
        "entityLANGUAGE",
        "entityDATE",
    ],
)


# --- Tuple of internal namespaces
_internal = SMGP, SMGO, SMGD


def is_internal_uri(val: Union[str, URIRef, Literal]) -> bool:
    """
    Returns whether a given value (string/rdflib.URIRef/rdflib.Literal) is an internal URI according to heritageconnector.namespace._internal.
    """

    return any([str(val).startswith(str(namespace)) for namespace in _internal])


def get_jsonld_context() -> List[dict]:
    """
    Get JSON-LD context for the whole namespace. Ignores namespaces in _internal.
    English language only.

    Returns:
        List[dict]. Example below.

    ```
    context = [
        {"@foaf": "http://xmlns.com/foaf/0.1/", "@language": "en"},
        {"@sdo": "https://schema.org/", "@language": "en"},
        {"@owl": "http://www.w3.org/2002/07/owl#", "@language": "en"},
        {"@xsd": "http://www.w3.org/2001/XMLSchema#", "@language": "en"},
        {"@wd": "http://www.wikidata.org/entity/", "@language": "en"},
        {"@wdt": "http://www.wikidata.org/prop/direct/", "@language": "en"},
        {"@prov": "http://www.w3.org/ns/prov#", "@language": "en"},
        {"@rdfs": "http://www.w3.org/2000/01/rdf-schema#", "@language": "en"},
        {"@skos": "http://www.w3.org/2004/02/skos/core#", "@language": "en"},
    ]
    ```

    """

    context = [
        {"@xsd": XSD.__str__(), "@language": "en"},
        {"@foaf": FOAF.__str__(), "@language": "en"},
        {"@owl": OWL.__str__(), "@language": "en"},
        {"@rdf": RDF.__str__(), "@language": "en"},
        {"@rdfs": RDFS.__str__(), "@language": "en"},
        {"@prov": PROV.__str__(), "@language": "en"},
        {"@sdo": SDO.__str__(), "@language": "en"},
        {"@skos": SKOS.__str__(), "@language": "en"},
        {"@wd": WD.__str__(), "@language": "en"},
        {"@wdt": WDT.__str__(), "@language": "en"},
    ]

    return context
