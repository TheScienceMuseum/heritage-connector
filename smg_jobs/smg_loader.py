import sys

sys.path.append("..")

from heritageconnector.config import config
from heritageconnector import datastore
import pandas as pd
from logging import getLogger
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import XSD, FOAF, OWL
from rdflib.serializer import Serializer
import json

logger = getLogger(__file__)

# Locatiom of CSV data to import
catalogue_data_path = config.MIMSY_CATALOGUE_PATH
people_data_path = config.MIMSY_PEOPLE_PATH
maker_data_path = config.MIMSY_MAKER_PATH
user_data_path = config.MIMSY_USER_PATH

collection = "SMG"
max_records = 1000

context = [
    {"@foaf": "http://xmlns.com/foaf/0.1/", "@language": "en"},
    {"@schema": "http://www.w3.org/2001/XMLSchema#", "@language": "en"},
    {"@owl": "http://www.w3.org/2002/07/owl#", "@language": "en"},
]


def load_object_data():
    """Load data from CSV files """

    catalogue_df = pd.read_csv(catalogue_data_path, low_memory=False, nrows=max_records)

    # Loop though CSV file and create/store records for each row
    # Note: We may want to optimise and send a bunch of new records to Elastic Search to process as a batch

    record_type = "object"
    for index, row in catalogue_df.iterrows():
        add_record(record_type, row)

    return


def load_people_and_orgs_data():
    """Load data from CSV files """

    people_df = pd.read_csv(people_data_path, low_memory=False, nrows=max_records)

    # Loop though CSV file and create/store records for each row
    # Note: We may want to optimise and send a bunch of new records to Elastic Search to process as a batch
    for index, row in people_df.iterrows():
        if row["GENDER"] == "M" or row["GENDER"] == "F":
            add_record("person", row)
        else:
            add_record("organisation", row)

    # We should use the isIndividual flag (but need to do a fresh export first)
    # for index, row in people_df.iterrows():
    #     if (row["GENDER"] == 'M' or row["GENDER"] == 'F'):
    #         add_record("person", row)
    #     else:
    #         add_record("organisation", row)
    return


def load_maker_data():
    """Load object -> maker -> people relationships from CSV files and add to existing records """

    maker_df = pd.read_csv(maker_data_path, low_memory=False, nrows=max_records)

    # Loop though CSV file and update exiting records for each row based on relationship value
    for index, row in maker_df.iterrows():
        obj = "https://collection.sciencemuseumgroup.org.uk/objects/co" + str(
            row["MKEY"]
        )
        maker = "https://collection.sciencemuseumgroup.org.uk/people/cp" + str(
            row["LINK_ID"]
        )
        relationship = (
            "maker"  # we may want to deal with other sub-classes of maker here later?
        )
        datastore.add_maker(obj, relationship, maker)

    return


def load_user_data():
    """Load object -> user -> people relationships from CSV files and add to existing records """

    user_df = pd.read_csv(user_data_path, low_memory=False, nrows=max_records)

    # Loop though CSV file and update exiting records for each row based on relationship value
    for index, row in user_df.iterrows():
        obj = "https://collection.sciencemuseumgroup.org.uk/objects/co" + str(
            row["MKEY"]
        )
        maker = "https://collection.sciencemuseumgroup.org.uk/people/cp" + str(
            row["LINK_ID"]
        )
        relationship = (
            "user"  # we may want to deal with other sub-types of user here later?
        )
        datastore.add_user(obj, relationship, maker)

    return


def add_record(record_type, row):
    """Create and store new HC record with a JSON-LD graph"""

    uri = ""
    if record_type == "object":
        uri = "https://collection.sciencemuseumgroup.org.uk/objects/co" + str(
            row["MKEY"]
        )
    if record_type == "person" or record_type == "organisation":
        # AdLib
        # uri = "https://collection.sciencemuseumgroup.org.uk/people/ap")
        # Mimsy
        uri = "https://collection.sciencemuseumgroup.org.uk/people/cp" + str(
            row["LINK_ID"]
        )

    data = {"uri": uri}
    jsonld = serialize_to_jsonld(record_type, uri, row)
    datastore.create(collection, record_type, data, jsonld)

    # print(data, jsonld)

    return


def serialize_to_jsonld(record_type, uri, row):
    """Returns a JSON-LD represention of a record"""

    g = Graph()
    record = URIRef(uri)

    # This code is effectivly the mapping from source data to the data we care about

    # ========================================================================
    # (1a) If the record is a person
    # http://xmlns.com/foaf/spec/#term_Person
    # ========================================================================
    if record_type == "person":

        # 1   PREFERRED_NAME    10 non-null     object
        # 2   TITLE_NAME        0 non-null      float64
        # 3   FIRSTMID_NAME     6 non-null      object
        # 4   LASTSUFF_NAME     10 non-null     object
        # 5   SUFFIX_NAME       0 non-null      float64
        # 6   HONORARY_SUFFIX   0 non-null      float64
        # 7   GENDER            10 non-null     object
        # 8   BRIEF_BIO         10 non-null     object
        # 9   DESCRIPTION       5 non-null      object
        # 10  NOTE              8 non-null      object
        # 11  BIRTH_DATE        6 non-null      object
        # 12  BIRTH_PLACE       8 non-null      object
        # 13  DEATH_DATE        6 non-null      object
        # 14  DEATH_PLACE       3 non-null      object
        # 15  CAUSE_OF_DEATH    2 non-null      object
        # 16  NATIONALITY       9 non-null      object
        # 17  OCCUPATION        10 non-null     object
        # 18  WEBSITE           0 non-null      float64
        # 19  AFFILIATION       0 non-null      float64
        # 20  LINGUISTIC_GROUP  0 non-null      float64
        # 21  TYPE              0 non-null      float64
        # 22  REFERENCE_NUMBER  0 non-null      float64
        # 23  SOURCE            10 non-null     object
        # 24  CREATE_DATE       10 non-null     object
        # 25  UPDATE_DATE

        # Add any personal details as FOAF / Scehema attributes
        if pd.notnull(row["PREFERRED_NAME"]):
            g.add((record, FOAF.givenName, Literal(row["PREFERRED_NAME"])))
        if pd.notnull(row["SUFFIX_NAME"]):
            g.add((record, FOAF.familyName, Literal(row["SUFFIX_NAME"])))
        if row["GENDER"] == "M":
            g.add((record, XSD.gender, Literal("Male")))
        if row["GENDER"] == "F":
            g.add((record, XSD.gender, Literal("Female")))
        # Need to convert to date format or 4 digit year to keep ElasticSearch happy
        # if pd.notnull(row["BIRTH_DATE"]):
        #     g.add((record, XSD.birthDate, Literal(row["BIRTH_DATE"])))
        # if pd.notnull(row["DEATH_DATE"]):
        #     g.add((record, XSD.deathDate, Literal(row["DEATH_DATE"])))
        if pd.notnull(row["BIRTH_PLACE"]):
            g.add((record, XSD.birthPlace, Literal(row["BIRTH_PLACE"])))
        if pd.notnull(row["DEATH_PLACE"]):
            g.add((record, XSD.deathPlace, Literal(row["DEATH_PLACE"])))
        if pd.notnull(row["OCCUPATION"]):
            g.add((record, XSD.ocupation, Literal(row["OCCUPATION"])))
        if pd.notnull(row["DESCRIPTION"]):
            g.add((record, XSD.description, Literal(row["DESCRIPTION"])))
        if pd.notnull(row["BRIEF_BIO"]):
            g.add((record, XSD.disambiguatingDescription, Literal(row["BRIEF_BIO"])))

    # ========================================================================
    # (1b) If the record is a organisation
    # http://xmlns.com/foaf/spec/#term_Organization
    # ========================================================================
    if record_type == "organisation":

        # Maybe we should use Agent rather than People/Orgs?
        # https://schema.org/agent
        # Add any personal details as FOAF / Schema attributes
        if pd.notnull(row["PREFERRED_NAME"]):
            g.add((record, FOAF.givenName, Literal(row["PREFERRED_NAME"])))
        if pd.notnull(row["DESCRIPTION"]):
            g.add((record, XSD.description, Literal(row["DESCRIPTION"])))
        if pd.notnull(row["BRIEF_BIO"]):
            g.add((record, XSD.disambiguatingDescription, Literal(row["BRIEF_BIO"])))

        next

    # ========================================================================
    # (2) If the record is a object
    # http://xmlns.com/foaf/spec/#term_SpatialThing
    # ========================================================================
    if record_type == "object":

        # 0   MKEY                  10 non-null     int64
        # 1   TITLE                 10 non-null     object
        # 2   ITEM_NAME             10 non-null     object
        # 3   CATEGORY1             10 non-null     object
        # 4   COLLECTOR             0 non-null      float64
        # 5   PLACE_COLLECTED       0 non-null      float64
        # 6   DATE_COLLECTED        0 non-null      float64
        # 7   PLACE_MADE            3 non-null      object
        # 8   CULTURE               0 non-null      float64
        # 9   DATE_MADE             3 non-null      object
        # 10  MATERIALS             5 non-null      object
        # 11  MEASUREMENTS          5 non-null      object
        # 12  EXTENT                0 non-null      float64
        # 13  DESCRIPTION           10 non-null     object
        # 14  ITEM_COUNT            10 non-null     int64
        # 15  PARENT_KEY            0 non-null      float64
        # 16  BROADER_TEXT          0 non-null      float64
        # 17  WHOLE_PART            10 non-null     object
        # 18  ARRANGEMENT           0 non-null      float64
        # 19  LANGUAGE_OF_MATERIAL  10 non-null     object

        if pd.notnull(row["TITLE"]):
            g.add((record, XSD.name, Literal(row["TITLE"])))
        if pd.notnull(row["DESCRIPTION"]):
            g.add((record, XSD.description, Literal(row["DESCRIPTION"])))
        if pd.notnull(row["ITEM_NAME"]):
            g.add((record, XSD.additionalType, Literal(row["ITEM_NAME"])))
        if pd.notnull(row["MATERIALS"]):
            materials = [[x.strip().lower() for x in str(row["MATERIALS"]).split(";")]]
            for material in materials:
                g.add((record, XSD.material, Literal(material)))

        # To add / what JSON-LD onotology do we use?
        # ------------------------------------------
        # ????DATE_MADE????
        # ????PLACE_MADE????
        # ????PLACE_COLLECTED????
        # ????DATE_COLLECTED????

        # if (row["CATEGORY1"]):
        #     g.add((record, XSD.category, Literal(row["CATEGORY1"])))
        # need to add a new quantitativeValue node for MEASUREMENTS
        # if (row["MEASUREMENTS"]):
        #     g.add((record, XSD.quantitativeValue, Literal(row["MEASUREMENTS"])))

        # Specific types ie. Artworks
        # https://schema.org/CreativeWork

        # g.add((record, FOAF.maker, URIRef('https://collection.sciencemuseumgroup.org.uk/objects/co8084947')

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

    return g.serialize(format="json-ld", context=context, indent=4).decode("utf-8")


# --------------------------------------------------------------------------------


if __name__ == "__main__":

    datastore.create_index()
    load_people_and_orgs_data()
    load_object_data()
    load_maker_data()
    load_user_data()
