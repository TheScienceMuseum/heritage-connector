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
non_graph_predicates = [
    XSD.description,
    SDO.disambiguatingDescription,
    # NOTE: enable the next two lines for KG embedding training (exclude first & last names)
    # WDT.P735, # first name
    # WDT.P734, # last name
]

mapping = {
    "PERSON": {
        "TITLE_NAME": {"RDF": FOAF.title},
        "FORENAME": {"RDF": FOAF.givenName},
        "SURNAME": {"RDF": FOAF.familyName},
        "NATURAL_NAME": {"RDF": RDFS.label},
        "BIRTHDATE_EARLIEST": {"RDF": SDO.birthDate},
        # "BIRTHDATE_LATEST": {"RDF": SDO.birthDate},
        "BIRTHPLACE": {"RDF": SDO.birthPlace},
        "DEATHPLACE": {"RDF": SDO.deathPlace},
        "NATIONALITY": {"RDF": SDO.nationality},
        "BIOGRAPHY": {"RDF": XSD.description},
        "DISAMBIGUATING_DESCRIPTION": {"RDF": SDO.disambiguatingDescription},
    },
    "ORGANISATION": {
        "DISPLAY_NAME": {"RDF": RDFS.label},
        "HISTORY": {"RDF": XSD.description},
        "FOUNDATION_PLACE_NAME": {"RDF": RDFS.label},
        "FOUNDATION_PLACE_ID": {"RDF": RDFS.label},
        "FOUNDATION_DATE_EARLIEST": {"RDF": SDO.foundingDate},
        # "FOUNDATION_DATE_LATEST": {"RDF": SDO.foundingDate},
        "DISAMBIGUATING_DESCRIPTION": {"RDF": SDO.disambiguatingDescription},
    },
    "OBJECT": {
        "PRIMARY_TITLE": {"RDF": RDFS.label},
        "PRIMARY_PLACE": {"RDF": SDO.place},
        "PRIMARY_DATE": {"RDF": SDO.dateCreated},
        "COMBINED_DESCRIPTION": {"RDF": XSD.description},
        # "DESCRIPTION": {"RDF": XSD.description},
        # "PHYS_DESCRIPTION": {"RDF": XSD.description},
        # "PRODUCTION_TYPE": {"RDF": XSD.description},
        "DISAMBIGUATING_DESCRIPTION": {"RDF": SDO.disambiguatingDescription},
        "OBJECT_TYPE": {"RDF": XSD.additionalType},
        "ACCESSION_NUMBER": {"WDT": WDT.P217},
        "COLLECTION": {"WDT": WDT.P195},
    },
    "EVENT": {
        "NAME": {"RDF": SDO.event},
        "DATE_EARLIEST": {"RDF": SDO.startDate},
        "DATE_LATEST": {"RDF": SDO.endDate},
    },
}
