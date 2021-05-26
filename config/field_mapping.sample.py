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
        "PREFERRED_NAME": {"RDF": RDFS.label},
        "FIRSTMID_NAME": {"RDF": FOAF.givenName},
        "LASTSUFF_NAME": {"RDF": FOAF.familyName},
        "GENDER": {"RDF": SDO.gender},
        # TODO: add date -> year guidance in docs
        "BIRTH_DATE": {"RDF": SDO.birthDate},
        "DEATH_DATE": {"RDF": SDO.deathDate},
        "BIRTH_PLACE": {"RDF": SDO.birthPlace},
        "DEATH_PLACE": {"RDF": SDO.deathPlace},
        "OCCUPATION": {"RDF": SDO.hasOccupation},
        "NATIONALITY": {"RDF": SDO.nationality},
        "BIOGRAPHY": {"RDF": XSD.description},
        "DISAMBIGUATING_DESCRIPTION": {"RDF": SDO.disambiguatingDescription},
    },
    "PERSON_ADLIB": {
        "PREFERRED_NAME": {"RDF": RDFS.label},
        "FIRSTMID_NAME": {"RDF": FOAF.givenName},
        "LASTSUFF_NAME": {"RDF": FOAF.familyName},
        "GENDER": {"RDF": SDO.gender},
        "BIRTH_DATE": {"RDF": SDO.birthDate},
        "DEATH_DATE": {"RDF": SDO.deathDate},
        "BIRTH_PLACE": {"RDF": SDO.birthPlace},
        "DEATH_PLACE": {"RDF": SDO.deathPlace},
        "NATIONALITY": {"RDF": SDO.nationality},
        "DESCRIPTION": {"RDF": XSD.description},
    },
    "ORGANISATION": {
        "PREFERRED_NAME": {"RDF": RDFS.label},
        "BIOGRAPHY": {"RDF": XSD.description},
        "DISAMBIGUATING_DESCRIPTION": {"RDF": SDO.disambiguatingDescription},
        "OCCUPATION": {"RDF": XSD.additionalType},
        "NATIONALITY": {"RDF": SDO.addressCountry},
        "BIRTH_DATE": {"RDF": SDO.foundingDate},
        "DEATH_DATE": {"RDF": SDO.dissolutionDate},
    },
    "ORGANISATION_ADLIB": {
        "LABEL": {"RDF": RDFS.label},
        "ALIAS": {"RDF": SKOS.altLabel},
        "BIRTH_DATE": {"RDF": SDO.foundingDate},
        "DEATH_DATE": {"RDF": SDO.dissolutionDate},
        "NATIONALITY": {"RDF": SDO.addressCountry},
        "DESCRIPTION": {"RDF": XSD.description},
    },
    "OBJECT": {
        "TITLE": {"RDF": RDFS.label},
        "DESCRIPTION": {"RDF": XSD.description},
        "DISAMBIGUATING_DESCRIPTION": {"RDF": SDO.disambiguatingDescription},
        "ITEM_NAME": {"RDF": XSD.additionalType},
        "MATERIALS": {"RDF": SDO.material},
        "DATE_MADE": {"RDF": SDO.dateCreated},
        "CATEGORY1": {"RDF": SDO.isPartOf},
    },
    "DOCUMENT": {
        "TITLE": {"RDF": RDFS.label},
        "DESCRIPTION": {"RDF": XSD.description},
        "SUBJECT": {"RDF": XSD.additionalType},
        "DATE_MADE": {"RDF": SDO.dateCreated},
    },
}
