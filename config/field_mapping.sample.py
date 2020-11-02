import sys

sys.path.append("..")

from heritageconnector.namespace import XSD, FOAF, OWL, RDF, RDFS, PROV, SDO, WD, WDT


mapping = {
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
        "BIRTH_PLACE": {"PID": WDT.P19, "RDF": XSD.birthPlace, "type": "location"},
        "DEATH_PLACE": {"PID": WDT.P20, "RDF": XSD.deathPlace, "type": "location"},
        "OCCUPATION": {
            "PID": WDT.P106,
            "RDF": SDO.hasOccupation,
            "type": "categorical",
        },
        "NATIONALITY": {"RDF": SDO.nationality, "type": "categorical"},
        "BIOGRAPHY": {"PID": "description", "RDF": XSD.description, "type": "str"},
        "NOTES": {"type": "str"},
    },
    "ORGANISATION": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "PREFERRED_NAME": {"RDF": RDFS.label, "type": "string"},
        "DESCRIPTION": {
            # "PID": "description",
            "RDF": XSD.description,
            "type": "str",
        },
        "BRIEF_BIO": {
            # "PID": "description",
            "RDF": XSD.disambiguatingDescription,
            "type": "str",
        },
        "OCCUPATION": {"RDF": XSD.additionalType, "type": "categorical"},
        "NATIONALITY": {"RDF": SDO.nationality, "type": "categorical"},
        "BIRTH_DATE": {"PID": WDT.P571, "RDF": SDO.foundingDate, "type": "numeric"},
        "DEATH_DATE": {"PID": WDT.P576, "RDF": SDO.dissolutionDate, "type": "numeric"},
    },
    "OBJECT": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "TITLE": {"RDF": RDFS.label, "type": "str"},
        "DESCRIPTION": {"PID": "description", "RDF": XSD.description, "type": "str"},
        "ITEM_NAME": {"PID": WDT.P31, "RDF": XSD.additionalType, "type": "list"},
        "MATERIALS": {"PID": WDT.P186, "RDF": XSD.material, "type": "list"},
        "DATE_MADE": {"PID": WDT.P571, "RDF": SDO.dateCreated, "type": "date"},
    },
    # NOTE: not being used at the moment
    "MAKERS": {
        # triples type has SUBJECT and OBJECT columns
        "TYPE": "triples",
        "PREDICATE": {"RDF": FOAF.maker},
    },
    "USERS": {"TYPE": "triples", "PREDICATE": {"RDF": PROV.used}},
}
