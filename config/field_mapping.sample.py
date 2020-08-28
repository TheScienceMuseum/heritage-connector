from rdflib.namespace import XSD, FOAF, OWL, SDO
from rdflib import Namespace

WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

mapping = {
    "PERSON": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "TITLE_NAME": {"RDF": FOAF.title},
        "PREFERRED_NAME": {"PID": "label", "RDF": XSD.name, "type": "string"},
        "FIRSTMID_NAME": {"PID": WDT.P735, "RDF": FOAF.givenName, "type": "string"},
        "LASTSUFF_NAME": {"PID": WDT.P734, "RDF": FOAF.familyName, "type": "string"},
        "GENDER": {"PID": WDT.P21, "RDF": XSD.gender, "type": "categorical"},
        # TODO: add date -> year guidance in docs
        "BIRTH_DATE": {"PID": WDT.P569, "RDF": XSD.birthDate, "type": "numeric"},
        "DEATH_DATE": {"PID": WDT.P570, "RDF": XSD.deathDate, "type": "numeric"},
        "BIRTH_PLACE": {"PID": WDT.P19, "RDF": XSD.birthPlace, "type": "location"},
        "DEATH_PLACE": {"PID": WDT.P20, "RDF": XSD.deathPlace, "type": "location"},
        "OCCUPATION": {"PID": WDT.P106, "RDF": XSD.occupation, "type": "categorical"},
        "NATIONALITY": {"RDF": SDO.nationality, "type": "categorical"},
        "BIOGRAPHY": {"PID": "description", "RDF": XSD.description, "type": "str"},
        "NOTES": {"type": "str"},
    },
    "ORGANISATION": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "PREFERRED_NAME": {"PID": "label", "RDF": FOAF.givenName, "type": "str"},
        "DESCRIPTION": {"PID": "description", "RDF": XSD.description, "type": "str"},
        "BRIEF_BIO": {
            "PID": "description",
            "RDF": XSD.disambiguatingDescription,
            "type": "str",
        },
        "OCCUPATION": {"RDF": XSD.occupation, "type": "list (str)"},
        "NATIONALITY": {"RDF": SDO.nationality, "type": "list (str)"},
        "BIRTH_DATE": {"PID": WDT.P571, "RDF": SDO.foundingDate, "type": "date"},
        "DEATH_DATE": {"PID": WDT.P576, "RDF": SDO.dissolutionDate, "type": "date"},
    },
    "OBJECT": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "TITLE": {"PID": "label", "RDF": XSD.name, "type": "str"},
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
    "USERS": {
        "TYPE": "triples",
        "PREDICATE": {
            # TODO: need to find RDF term for user
            "RDF": FOAF.knows
        },
    },
}
