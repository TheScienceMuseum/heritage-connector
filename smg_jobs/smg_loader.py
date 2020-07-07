import sys

sys.path.append("..")
from heritageconnector import datastore
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import XSD, FOAF, OWL
from rdflib.serializer import Serializer
from pprint import pprint

collection = "SMG"
context = [
    {"@foaf": "http://xmlns.com/foaf/0.1/", "@language": "en"},
    {"@schema": "http://www.w3.org/2001/XMLSchema#", "@language": "en"},
    {"@owl": "http://www.w3.org/2002/07/owl#", "@language": "en"},
]


def loadObjectData(csv):
    """Load data from CSV files """

    # loop though CSV file and create/store records for each row
    # Note: We may want to optimise and send a bunch of new records to Elastic Search to process as a batch
    data = {}
    record_type = "object"
    addRecord(record_type, data)

    return


def loadPeopleData(csv):
    """Load data from CSV files """

    # loop though CSV file and create/store records for each row
    # Note: We may want to optimise and send a bunch of new records to Elastic Search to process as a batch
    data = {}
    record_type = "people"
    addRecord(record_type, data)

    return


def loadMakerData(csv):
    """Load object -> maker -> people relationships from CSV files and add to existing records """

    # loop though CSV file and update exiting records for each row based on relationship value

    # s = subject = id
    # p = predicate = relationship
    # o = object = thing we are linking to

    s = ""
    p = "MADE"  # should this be the RDF / Wikidata value?
    o = "https://collection.sciencemuseumgroup.org.uk/objects/co146411"
    addRelationship(s, p, o)

    return


def loadUserData(csv):
    """Load object -> user -> people relationships from CSV files and add to existing records """

    # loop though CSV file and update exiting records for each row based on 'predicate' value

    # s = subject = id
    # p = predicate = relationship
    # o = object = thing we are linking to

    s = ""
    p = "USED"  # should this be the RDF / Wikidata value? What about more granular values?
    o = "https://collection.sciencemuseumgroup.org.uk/people/cp37182"
    addRelationship(s, p, o)

    return


def addRecord(collection, record_type, data):
    """Create and store new HC record with a JSON-LD graph"""

    # Question: do we want to massage/format some 'additional' data (outside of the JSON-LD)
    # like the collectionName to be stored on the Elastic record?
    jsonld = serializeToJsonld(record_type, data)
    datastore.create(collection, record_type, data, jsonld)

    return


def addRelationship(s, p, o):
    """Add a new RDF relationship to an an existing record"""

    # s = subject = id
    # p = predicate = relationship
    # o = object = thing we are linking to

    # Do we do the lookup agaisnt out config file here? (I think yes)
    # Do we store multiple entries for both Wikidata and RDF? (I think yes)

    record = datastore.get(id)
    g = Graph().parse(data=record, format="json-ld")

    g.add(
        (
            URIRef(s),
            URIRef(
                p
            ),  # we may need to convert our Wikidata P values and RDF verbs to URLs here
            URIRef(o),
        )
    )

    record.graph = g.serialize(format="json-ld", context=context, indent=4).decode(
        "utf-8"
    )

    datastore.update(id, record)

    return


def serializeToJsonld(record_type, record):
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

    # objects they made
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
    g.add((record, OWL.sameAs, URIRef("https://www.wikidata.org/wiki/1000")))

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

    datastore.createIndex()
    jsonld = addRecord("SMG", "object", data)
    pprint(jsonld)
