import sys

sys.path.append("..")

from heritageconnector.config import config, field_mapping
from heritageconnector import datastore
from heritageconnector.utils.data_transformation import get_year_from_date_value
import pandas as pd
from logging import getLogger
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import XSD, FOAF, OWL
from rdflib.serializer import Serializer
import json

logger = getLogger(__file__)
max_records = 1000

#  =============== LOADING SMG DATA ===============
# Location of CSV data to import
catalogue_data_path = config.MIMSY_CATALOGUE_PATH
people_data_path = config.MIMSY_PEOPLE_PATH
maker_data_path = config.MIMSY_MAKER_PATH
user_data_path = config.MIMSY_USER_PATH

collection = "SMG"

context = [
    {"@foaf": "http://xmlns.com/foaf/0.1/", "@language": "en"},
    {"@schema": "http://schema.org/", "@language": "en"},
    {"@owl": "http://www.w3.org/2002/07/owl#", "@language": "en"},
    {"@xsd": "http://www.w3.org/2001/XMLSchema#", "@language": "en"},
    {"@wd": "http://www.wikidata.org/entity/", "@language": "en"},
    {"@wdt": "http://www.wikidata.org/prop/direct/", "@language": "en"},
]

collection_prefix = "https://collection.sciencemuseumgroup.org.uk/objects/co"
people_prefix = "https://collection.sciencemuseumgroup.org.uk/people/cp"


def load_object_data():
    """Load data from CSV files """

    table_name = "OBJECT"
    catalogue_df = pd.read_csv(catalogue_data_path, low_memory=False, nrows=max_records)
    catalogue_df = catalogue_df.rename(columns={"MKEY": "ID"})
    catalogue_df["PREFIX"] = collection_prefix
    catalogue_df["MATERIALS"] = catalogue_df["MATERIALS"].apply(
        lambda i: [x.strip().lower() for x in str(i).replace(";", ",").split(",")]
    )
    catalogue_df["ITEM_NAME"] = catalogue_df["ITEM_NAME"].apply(
        lambda i: [x.strip().lower() for x in str(i).replace(";", ",").split(",")]
    )

    # Loop though CSV file and create/store records for each row
    # Note: We may want to optimise and send a bunch of new records to Elastic Search to process as a batch

    for dummy, row in catalogue_df.iterrows():
        add_record(table_name, row)

    return


def load_people_data():
    """Load data from CSV files """

    # identifier in field_mapping
    table_name = "PERSON"

    people_df = pd.read_csv(people_data_path, low_memory=False, nrows=max_records)
    # TODO: use isIndividual flag here
    people_df = people_df[people_df["GENDER"].isin(["M", "F"])]

    # PREPROCESS
    people_df = people_df.rename(columns={"LINK_ID": "ID"})
    people_df["PREFIX"] = people_prefix
    people_df["BIRTH_DATE"] = people_df["BIRTH_DATE"].apply(get_year_from_date_value)
    people_df["DEATH_DATE"] = people_df["DEATH_DATE"].apply(get_year_from_date_value)
    people_df["OCCUPATION"] = people_df["OCCUPATION"].apply(
        lambda i: [x.strip().lower() for x in str(i).replace(";", ",").split(",")]
    )
    people_df.loc[:, "BIOGRAPHY"] = people_df.loc[:, "DESCRIPTION"]
    people_df.loc[:, "NOTES"] = (
        str(people_df.loc[:, "DESCRIPTION"]) + " \n " + str(people_df.loc[:, "NOTE"])
    )
    # TODO: map gender to Wikidata QIDs

    # TODO: use Elasticsearch batch mechanism for loading
    for _, row in people_df.iterrows():
        add_record(table_name, row)


def load_orgs_data():
    # identifier in field_mapping
    table_name = "ORGANISATION"

    org_df = pd.read_csv(people_data_path, low_memory=False, nrows=max_records)
    # TODO: use isIndividual flag here
    org_df = org_df[org_df["GENDER"] == "N"]

    # PREPROCESS
    org_df = org_df.rename(columns={"LINK_ID": "ID"})
    org_df["PREFIX"] = people_prefix

    # TODO: use Elasticsearch batch mechanism for loading
    for _, row in org_df.iterrows():
        add_record(table_name, row)

    return


def load_maker_data():
    """Load object -> maker -> people relationships from CSV files and add to existing records """
    # identifier in field mapping
    maker_df = pd.read_csv(maker_data_path, low_memory=False, nrows=max_records)

    maker_df["MKEY"] = collection_prefix + maker_df["MKEY"].astype(str)
    maker_df["LINK_ID"] = people_prefix + maker_df["LINK_ID"].astype(str)
    maker_df = maker_df.rename(columns={"MKEY": "SUBJECT", "LINK_ID": "OBJECT"})

    for _, row in maker_df.iterrows():
        datastore.update_graph(row["SUBJECT"], FOAF.maker, row["OBJECT"])

    return


def load_user_data():
    """Load object -> user -> people relationships from CSV files and add to existing records """
    user_df = pd.read_csv(user_data_path, low_memory=False, nrows=max_records)

    user_df["MKEY"] = collection_prefix + user_df["MKEY"].astype(str)
    user_df["LINK_ID"] = people_prefix + user_df["LINK_ID"].astype(str)
    user_df = user_df.rename(columns={"MKEY": "SUBJECT", "LINK_ID": "OBJECT"})

    for _, row in user_df.iterrows():
        datastore.update_graph(row["SUBJECT"], FOAF.knows, row["OBJECT"])

    return


#  =============== GENERIC FUNCTIONS FOR LOADING (move these?) ===============


def add_record(table_name, row):
    """Create and store new HC record with a JSON-LD graph"""

    uri_prefix = row["PREFIX"]
    uri = uri_prefix + str(row["ID"])

    data = {"uri": uri}
    jsonld = serialize_to_jsonld(table_name, uri, row)

    datastore.create(collection, table_name, data, jsonld)

    return


def serialize_to_jsonld(table_name: str, uri: str, row: pd.Series):
    """Returns a JSON-LD represention of a record"""

    g = Graph()
    record = URIRef(uri)

    # This code is effectivly the mapping from source data to the data we care about
    table_mapping = field_mapping.mapping[table_name]

    keys = {
        k for k, v in table_mapping.items() if k not in ["ID", "PREFIX"] and "RDF" in v
    }

    for col in keys:
        # this will trigger for the first row in the dataframe
        #  TODO: put this in a separate checker function that checks each table against config on loading
        if col not in row.index:
            raise KeyError(f"column {col} not in data for table {table_name}")

        if bool(row[col]) and (str(row[col]) != "nan"):
            if isinstance(row[col], list):
                [
                    g.add((record, table_mapping[col]["RDF"], Literal(val)))
                    for val in row[col]
                    if str(val) != "nan"
                ]
            else:
                g.add((record, table_mapping[col]["RDF"], Literal(row[col])))

    return g.serialize(format="json-ld", context=context, indent=4).decode("utf-8")


if __name__ == "__main__":

    datastore.create_index()
    load_people_data()
    load_orgs_data()
    load_object_data()
    load_maker_data()
    load_user_data()
