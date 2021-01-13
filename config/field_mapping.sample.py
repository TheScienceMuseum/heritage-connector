import sys

sys.path.append("..")

from heritageconnector.namespace import (
    XSD,
    FOAF,
    OWL,
    RDF,
    RDFS,
    PROV,
    SDO,
    WD,
    WDT,
    SKOS,
)

# PIDs to store in ES in _source.data rather than _source.graph. You may want to do this to keep the graph small for analytics purposes,
# whilst keeping some useful information in the Elasticsearch index.
non_graph_pids = [
    "description",
    # NOTE: enable the next two lines for KG embedding training (exclude first & last names)
    # WDT.P735, # first name
    # WDT.P734, # last name
]


mapping = {
    "PERSON_ADLIB": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "TITLE_NAME": {"RDF": FOAF.title},
        "PREFERRED_NAME": {"RDF": RDFS.label, "type": "string"},
        "FIRSTMID_NAME": {"PID": WDT.P735, "RDF": FOAF.givenName, "type": "string"},
        "LASTSUFF_NAME": {"PID": WDT.P734, "RDF": FOAF.familyName, "type": "string"},
        "GENDER": {
            "PID": WDT.P21,
            "RDF": SDO.gender,
            "type": "categorical",
            "wikidata_entity": True,
        },
        # TODO: add date -> year guidance in docs
        "BIRTH_DATE": {"PID": WDT.P569, "RDF": SDO.birthDate, "type": "numeric"},
        "DEATH_DATE": {"PID": WDT.P570, "RDF": SDO.deathDate, "type": "numeric"},
        "BIRTH_PLACE": {"PID": WDT.P19, "RDF": SDO.birthPlace, "type": "location"},
        "DEATH_PLACE": {"PID": WDT.P20, "RDF": SDO.deathPlace, "type": "location"},
        # "OCCUPATION": {
        #     "PID": WDT.P106,
        #     "RDF": SDO.hasOccupation,
        #     "type": "categorical",
        # },
        "NATIONALITY": {"RDF": SDO.nationality, "type": "categorical"},
    },
    "ORGANISATION_ADLIB": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "PREFERRED_NAME": {"RDF": RDFS.label, "type": "string"},
        "DESCRIPTION": {
            # "PID": "description",
            "RDF": XSD.description,
            "type": "str",
        },
        "NATIONALITY": {"RDF": SDO.addressCountry, "type": "categorical"},
        "BIRTH_DATE": {"PID": WDT.P571, "RDF": SDO.foundingDate, "type": "numeric"},
        "DEATH_DATE": {"PID": WDT.P576, "RDF": SDO.dissolutionDate, "type": "numeric"},
    },
    "DOCUMENT_ADLIB": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "TITLE": {"RDF": RDFS.label, "type": "str"},
        "DESCRIPTION": {"PID": "description", "RDF": XSD.description, "type": "str"},
        "SUBJECT": {"PID": WDT.P31, "RDF": XSD.additionalType, "type": "list"},
        "DATE_MADE": {"PID": WDT.P571, "RDF": SDO.dateCreated, "type": "date"},
    },
    "PERSON": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "TITLE_NAME": {"RDF": FOAF.title},
        "PREFERRED_NAME": {"RDF": RDFS.label, "type": "string"},
        "FIRSTMID_NAME": {"PID": WDT.P735, "RDF": FOAF.givenName, "type": "string"},
        "LASTSUFF_NAME": {"PID": WDT.P734, "RDF": FOAF.familyName, "type": "string"},
        "GENDER": {
            "PID": WDT.P21,
            "RDF": SDO.gender,
            "type": "categorical",
            "wikidata_entity": True,
        },
        # TODO: add date -> year guidance in docs
        "BIRTH_DATE": {"PID": WDT.P569, "RDF": SDO.birthDate, "type": "numeric"},
        "DEATH_DATE": {"PID": WDT.P570, "RDF": SDO.deathDate, "type": "numeric"},
        "BIRTH_PLACE": {"PID": WDT.P19, "RDF": SDO.birthPlace, "type": "location"},
        "DEATH_PLACE": {"PID": WDT.P20, "RDF": SDO.deathPlace, "type": "location"},
        "OCCUPATION": {
            "PID": WDT.P106,
            "RDF": SDO.hasOccupation,
            "type": "categorical",
        },
        "NATIONALITY": {"RDF": SDO.nationality, "type": "categorical"},
        "BIOGRAPHY": {"PID": "description", "RDF": XSD.description, "type": "str"},
        "NOTES": {"type": "str"},
        "adlib_id": {"RDF": FOAF.page},
        "adlib_ALIAS": {"RDF": SKOS.altLabel},
    },
    "ORGANISATION": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "PREFERRED_NAME": {"RDF": RDFS.label, "type": "string"},
        "DESCRIPTION": {"PID": "description", "RDF": XSD.description, "type": "str"},
        "OCCUPATION": {"RDF": XSD.additionalType, "type": "categorical"},
        "NATIONALITY": {"RDF": SDO.addressCountry, "type": "categorical"},
        "BIRTH_DATE": {"PID": WDT.P571, "RDF": SDO.foundingDate, "type": "numeric"},
        "DEATH_DATE": {"PID": WDT.P576, "RDF": SDO.dissolutionDate, "type": "numeric"},
        "adlib_id": {"RDF": FOAF.page},
        "adlib_ALIAS": {"RDF": SKOS.altLabel},
    },
    "OBJECT": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "TITLE": {"RDF": RDFS.label, "type": "str"},
        "DESCRIPTION": {"PID": "description", "RDF": XSD.description, "type": "str"},
        "ITEM_NAME": {"PID": WDT.P31, "RDF": XSD.additionalType, "type": "list"},
        "MATERIALS": {"PID": WDT.P186, "RDF": SDO.material, "type": "list"},
        "DATE_MADE": {"PID": WDT.P571, "RDF": SDO.dateCreated, "type": "date"},
        "CATEGORY1": {"RDF": SDO.isPartOf, "type": "categorical"},
    },
    # NOTE: not being used at the moment
    "MAKERS": {
        # triples type has SUBJECT and OBJECT columns
        "TYPE": "triples",
        "PREDICATE": {"RDF": FOAF.maker},
    },
    "USERS": {"TYPE": "triples", "PREDICATE": {"RDF": PROV.used}},
}
