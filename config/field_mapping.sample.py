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
    HC,
)


# PIDs to store in ES in _source.data rather than _source.graph. You may want to do this to keep the graph small for analytics purposes,
# whilst keeping some useful information in the Elasticsearch index.
non_graph_predicates = [
    XSD.description,
    SDO.disambiguatingDescription,
    # NOTE: enable the next two lines for KG embedding training (exclude first & last names)
    # WDT.P735, # first name
    # WDT.P734, # last name
    # For blog/journal:
    SDO.abstract,
    SDO.text,
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
        "DATABASE": {"RDF": HC.database},
    },
    "ORGANISATION": {
        "PREFERRED_NAME": {"RDF": RDFS.label},
        "BIOGRAPHY": {"RDF": XSD.description},
        "DISAMBIGUATING_DESCRIPTION": {"RDF": SDO.disambiguatingDescription},
        "OCCUPATION": {"RDF": XSD.additionalType},
        "NATIONALITY": {"RDF": SDO.addressCountry},
        "BIRTH_DATE": {"RDF": SDO.foundingDate},
        "DEATH_DATE": {"RDF": SDO.dissolutionDate},
        "DATABASE": {"RDF": HC.database},
    },
    "OBJECT": {
        "TITLE": {"RDF": RDFS.label},
        "DESCRIPTION": {"RDF": XSD.description},
        "DISAMBIGUATING_DESCRIPTION": {"RDF": SDO.disambiguatingDescription},
        "ITEM_NAME": {"RDF": XSD.additionalType},
        "MATERIALS": {"RDF": SDO.material},
        "DATE_MADE": {"RDF": SDO.dateCreated},
        "CATEGORY1": {"RDF": SDO.isPartOf},
        "DATABASE": {"RDF": HC.database},
    },
    "DOCUMENT": {
        "TITLE": {"RDF": RDFS.label},
        "DESCRIPTION": {"RDF": XSD.description},
        "DISAMBIGUATING_DESCRIPTION": {"RDF": SDO.disambiguatingDescription},
        "SUBJECT": {"RDF": XSD.additionalType},
        "DATE_MADE": {"RDF": SDO.dateCreated},
        "DATABASE": {"RDF": HC.database},
    },
    "BLOG_POST": {
        "author": {"RDF": SDO.author},
        "date": {"RDF": SDO.dateCreated},
        "title": {"RDF": RDFS.label},
        "caption": {"RDF": SDO.abstract},
        "categories": {"RDF": SDO.genre},
        "tags": {"RDF": SDO.keywords},
        "text_by_paragraph": {"RDF": SDO.text},
        "links": {"RDF": SDO.mentions},
    },
    "JOURNAL_ARTICLE": {
        "doi": {"RDF": SDO.identifier},
        "author": {"RDF": SDO.author},
        "title": {"RDF": RDFS.label},
        "issue": {"RDF": SDO.isPartOf},
        "keywords": {"RDF": SDO.genre},
        "tags": {"RDF": SDO.keywords},
        "text_by_paragraph": {"RDF": SDO.text},
    },
}
