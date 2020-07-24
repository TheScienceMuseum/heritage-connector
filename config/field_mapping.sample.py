from rdflib.namespace import XSD, FOAF, OWL
from rdflib import Namespace

WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

# Objects
# ----------------
# 1   TITLE                 XSD.name               string
# 2   OBJECT_TYPE (ITEM_NAME)             XSD.additionalType     cslist
# 2b  OBJECT_TYPE_WIKICODE # maybe a new field to hold the Wikidata ID for it is avaliable
# 3   CATEGORY (CATEGORY1): "Space"
# 3b  CATEGORY__WIKICODE # maybe a new field to hold the Wikidata ID for it is avaliable
# 4   COLLECTOR
# 5   PLACE_COLLECTED
# 6   DATE_COLLECTED
# 7   PLACE_MADE
# 8   CULTURE
# 9   DATE_MADE
# 10  MATERIALS             XSD.material            cslist
# 10b MATERIALS__WIKICODE # maybe a new field to hold the Wikidata ID for it is avaliable
# 11  MEASUREMENTS
# 12  EXTENT
# 13  DESCRIPTION           XSD.description         string

# People
# ----------------
# 1   PREFERRED_NAME
# 2   TITLE_NAME
# 3   FIRSTMID_NAME     FOAF.givenName          string
# 4   LASTSUFF_NAME     FOAF.familyName         string
# 5   SUFFIX_NAME
# 6   HONORARY_SUFFIX
# 7   GENDER            XSD.gender              M/F
# 8   BRIEF_BIO         XSD.disambiguatingDescription string
# 9   DESCRIPTION       XSD.description         string
# 10  NOTE
# 11  BIRTH_DATE        XSD.birthDate          dateString
# 12  BIRTH_PLACE       XSD.birthPlace          string
# 13  DEATH_DATE        XSD.birthDate          dateString
# 14  DEATH_PLACE       XSD.deathPlace          string
# 15  CAUSE_OF_DEATH    WDT.P509
# 16  NATIONALITY
# 17  OCCUPATION
# 18  WEBSITE
# 19  AFFILIATION
# 20  LINGUISTIC_GROUP
# 21  TYPE
# 22  REFERENCE_NUMBER
# 23  SOURCE
# 24  CREATE_DATE
# 25  UPDATE_DATE

# TODO:
# ========================================================================
# (3) If the record is an archive document
# --------------------------------
# if (record_type == "document"):
#     next

# ========================================================================
# (4) If the record is an article (editorial/blog post)
# ========================================================================
# if (record_type == "article"):
#     next


mapping = {
    "PERSON": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "PREFERRED_NAME": {"PID": "label", "RDF": XSD.name, "type": "str"},
        "FIRSTMID_NAME": {"PID": WDT.P735, "RDF": FOAF.givenName, "type": "str"},
        "LASTSUFF_NAME": {"PID": WDT.P734, "RDF": FOAF.familyName, "type": "str"},
        "GENDER": {"PID": WDT.P21, "RDF": XSD.gender, "type": "wd_entity"},
        # TODO: add date -> year guidance in docs
        "BIRTH_DATE": {"PID": WDT.P569, "RDF": XSD.birthDate, "type": "date"},
        "DEATH_DATE": {"PID": WDT.P570, "RDF": XSD.deathDate, "type": "date"},
        "BIRTH_PLACE": {"PID": WDT.P19, "RDF": XSD.birthPlace, "type": "place"},
        "DEATH_PLACE": {"PID": WDT.P20, "RDF": XSD.deathPlace, "type": "place"},
        # TODO: add list formatting guidance in docs
        "OCCUPATION": {"PID": WDT.P106, "RDF": XSD.occupation, "type": "list (str)"},
        # TODO: combine description & note to refer to Wikidata description field
        "BIOGRAPHY": {"type": "str", "RDF": XSD.description},
        "NOTES": {"type": "str"},
    },
    "ORGANISATION": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "PREFERRED_NAME": {"PID": "label", "RDF": FOAF.givenName, "type": "str"},
        "DESCRIPTION": {"PID": "description", "RDF": XSD.description, "type": "str"},
        "BRIEF_BIO": {"PID": "", "RDF": XSD.disambiguatingDescription, "type": "str"},
    },
    "OBJECT": {
        "ID": {"type": "index"},
        # TODO: make PREFIX field when loading data
        "PREFIX": {"type": "prefix"},
        "TITLE": {"PID": "label", "RDF": XSD.name, "type": "str"},
        "DESCRIPTION": {"PID": "description", "RDF": XSD.description, "type": "str"},
        "ITEM_NAME": {"PID": WDT.P31, "RDF": XSD.additionalType, "type": "list"},
        "MATERIALS": {"PID": WDT.P186, "RDF": XSD.material, "type": "list"},
    },
}
