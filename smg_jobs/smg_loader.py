import sys

sys.path.append("..")
from heritageconnector import datastore
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import XSD, FOAF, OWL
from rdflib.serializer import Serializer
from pprint import pprint

collectionName = "SMG"


def load_object_data(csv):
    """Load data from CSV files """

    # loop though CSV file and create/store records for each row
    # Note: We may want to optimise and send a bunch of new records to Elastic Search to process as a batch
    create(data)

    return


def load_maker_data(csv):
    """Load object -> user -> people relationships from CSV files and add to existing records """

    # loop though CSV file and update exiting records for each row based on 'predicate' value
    id = ""
    triple = ("", "", "")
    update(id, triple)

    return


def load_user_data(csv):
    """Load object -> maker -> people relationships from CSV files and add to existing records """

    # loop though CSV file and update exiting records for each row based on 'predicate' value
    id = ""
    triple = ""
    update(id, triple)

    return


def create(collectionName, data):
    """Create and store new HC record with a JSON-LD graph"""

    # Question: do we want to massage/format some 'additional' data (outside of the JSON-LD)
    # like the collectionName to be stored on the Elastic record?
    jsonld = serialize(data)
    datastore.create(data, jsonld)

    return


def update(id, triple):
    """Add a new triple of info to an an existing record"""

    # Used to add object -> user -> people relationships type loaded seperatly
    # Question: How do we pass this info? JSON-LD, internal RDFLib format or just a S->P->O tuple?
    # Question: Or do we load the current record as JSON-LD and update it here? Seems inefficent?
    datastore.update(id, triple)

    return


def serialize(data):
    """Returns a JSON-LD represention of a record"""

    g = Graph()
    record = URIRef(data["uri"])

    # This code is effectivly the mapping from source data to the data we care about

    # If the record is a person
    # http://xmlns.com/foaf/spec/#term_Person
    # --------------------------------

    # Add any personal details as FOAF / Scehema attributes
    g.add((record, FOAF.givenName, Literal(data["given_name"])))
    g.add((record, FOAF.familyName, Literal(data["family_name"])))
    g.add((record, XSD.birthDate, Literal(data["birth_date"])))
    g.add((record, XSD.deathDate, Literal(data["death_date"])))

    # objectc they made
    g.add(
        (
            record,
            FOAF.made,
            URIRef("https://collection.sciencemuseumgroup.org.uk/objects/co8084947"),
        )
    )

    # Add any wikidata URIs using OWL:sameAs
    g.add(
        (
            record,
            OWL.sameAs,
            URIRef("https://www.wikidata.org/wiki/" + data["wikidata"]),
        )
    )

    # if the record is a org
    # --------------------------------
    # http://xmlns.com/foaf/spec/#term_Organization

    # if the record is a object
    # http://xmlns.com/foaf/spec/#term_SpatialThing
    # --------------------------------
    # g.add((record, FOAF.maker, URIRef('https://collection.sciencemuseumgroup.org.uk/objects/co8084947')

    # if the record is an archive document
    # --------------------------------

    # if the record is an article (editorial/blog post)
    # --------------------------------

    context = [
        {"@foaf": "http://xmlns.com/foaf/0.1/", "@language": "en"},
        {"@schema": "http://www.w3.org/2001/XMLSchema#", "@language": "en"},
        {"@owl": "http://www.w3.org/2002/07/owl#", "@language": "en"},
    ]
    return g.serialize(format="json-ld", context=context, indent=4).decode("utf-8")


# ----------------------------------------------

if __name__ == "__main__":

    data = {
        "uri": "https://collection.sciencemuseumgroup.org.uk/people/cp36993/charles-babbage",
        "name": "Charles Babbage",
        "family_name": "Babbage",
        "given_name": "Charles",
        "birth_date": "1800/01/01",
        "death_date": "1850/01/01",
        "born": "London",
        "wikidata": "Q46633",
    }

    datastore.create_index()
    jsonld = create("SMG", data)
    pprint(jsonld)
