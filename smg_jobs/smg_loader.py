import sys

sys.path.append("..")

from heritageconnector.config import config, field_mapping
from heritageconnector import datastore
from heritageconnector.utils.data_transformation import get_year_from_date_value
from heritageconnector.utils.wikidata import qid_to_url
import pandas as pd
from logging import getLogger
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import XSD, FOAF, OWL
from rdflib.serializer import Serializer
import json
import os
from tqdm.auto import tqdm

# disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

logger = getLogger(__file__)

# set to None for no limit
max_records = None

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


def process_text(text: str):
    """
    Remove newlines/other problematic characters
    """
    newstr = str(text)
    newstr = newstr.replace("\n", " ")
    # newstr = newstr.replace("\r\n", " ")
    # newstr = newstr.replace("\r", " ")
    newstr = newstr.replace("\t", " ")

    return newstr


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
    catalogue_df["DESCRIPTION"] = catalogue_df["DESCRIPTION"].apply(process_text)

    print("loading object data")
    add_records(table_name, catalogue_df)

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
    # remove newlines and tab chars
    people_df.loc[:, "DESCRIPTION"] = people_df.loc[:, "DESCRIPTION"].apply(
        process_text
    )
    people_df.loc[:, "NOTE"] = people_df.loc[:, "NOTE"].apply(process_text)
    # create combined text fields
    people_df.loc[:, "BIOGRAPHY"] = people_df.loc[:, "DESCRIPTION"]
    people_df.loc[:, "NOTES"] = (
        str(people_df.loc[:, "DESCRIPTION"]) + " " + str(people_df.loc[:, "NOTE"])
    )
    # TODO: map gender to Wikidata QIDs

    print("loading people data")
    add_records(table_name, people_df)


def load_orgs_data():
    # identifier in field_mapping
    table_name = "ORGANISATION"

    org_df = pd.read_csv(people_data_path, low_memory=False, nrows=max_records)
    # TODO: use isIndividual flag here
    org_df = org_df[org_df["GENDER"] == "N"]

    # PREPROCESS
    org_df = org_df.rename(columns={"LINK_ID": "ID"})
    org_df["PREFIX"] = people_prefix

    org_df["DESCRIPTION"] = org_df["DESCRIPTION"].apply(process_text)
    org_df["BRIEF_BIO"] = org_df["BRIEF_BIO"].apply(process_text)

    # TODO: use Elasticsearch batch mechanism for loading
    print("loading orgs data")
    add_records(table_name, org_df)

    return


def load_maker_data():
    """Load object -> maker -> people relationships from CSV files and add to existing records """
    # identifier in field mapping
    maker_df = pd.read_csv(maker_data_path, low_memory=False, nrows=max_records)

    maker_df["MKEY"] = collection_prefix + maker_df["MKEY"].astype(str)
    maker_df["LINK_ID"] = people_prefix + maker_df["LINK_ID"].astype(str)
    maker_df = maker_df.rename(columns={"MKEY": "SUBJECT", "LINK_ID": "OBJECT"})

    print("loading maker data")
    add_triples(maker_df, FOAF.maker, subject_col="SUBJECT", object_col="OBJECT")

    return


def load_user_data():
    """Load object -> user -> people relationships from CSV files and add to existing records """
    user_df = pd.read_csv(user_data_path, low_memory=False, nrows=max_records)

    user_df["MKEY"] = collection_prefix + user_df["MKEY"].astype(str)
    user_df["LINK_ID"] = people_prefix + user_df["LINK_ID"].astype(str)
    user_df = user_df.rename(columns={"MKEY": "SUBJECT", "LINK_ID": "OBJECT"})

    print("loading user data")
    add_triples(user_df, FOAF.knows, subject_col="SUBJECT", object_col="OBJECT")

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


def add_records(table_name, df):
    """Use ES parallel_bulk mechanism to add records from a table"""
    generator = record_create_generator(table_name, df)
    datastore.es_bulk(generator, len(df))


def record_create_generator(table_name, df):
    """Yields jsonld for a row for use with ES bulk helpers"""

    for _, row in df.iterrows():
        uri_prefix = row["PREFIX"]
        uri = uri_prefix + str(row["ID"])

        jsonld = serialize_to_jsonld(table_name, uri, row)

        doc = {
            "_id": uri,
            "uri": uri,
            "collection": collection,
            "type": table_name,
            "graph": json.loads(jsonld),
        }

        yield doc


def add_triples(df, predicate, subject_col="SUBJECT", object_col="OBJECT"):
    """Add triples with RDF predicate and dataframe containing subject and object columns"""

    generator = record_update_generator(df, predicate, subject_col, object_col)
    datastore.es_bulk(generator, len(df))


def record_update_generator(df, predicate, subject_col="SUBJECT", object_col="OBJECT"):
    """Yields jsonld docs to update existing records with new triples"""

    for _, row in df.iterrows():
        g = Graph()
        g.add((URIRef(row[subject_col]), predicate, URIRef(row[object_col])))

        jsonld_dict = json.loads(
            g.serialize(format="json-ld", context=context, indent=4)
        )
        _ = jsonld_dict.pop("@id")
        _ = jsonld_dict.pop("@context")

        body = {"graph": jsonld_dict}

        doc = {"_id": row[subject_col], "_op_type": "update", "doc": body}

        yield doc


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


def load_sameas_people_orgs(pickle_path):
    """
    pickle_path points to a dataframe with a 'qcodes_filtered' column containing exact qcode matches for items
    """

    if os.path.exists(pickle_path):
        df = pd.read_pickle(pickle_path)

        df_links = df[df["qcodes_filtered"].apply(len) == 1]
        df_links.loc[:, "QID"] = df_links.loc[:, "qcodes_filtered"].apply(
            lambda i: i[0]
        )
        df_links = df_links[["LINK_ID", "QID"]]

        # transform IDs to URLs
        df_links["LINK_ID"] = df_links["LINK_ID"].apply(
            lambda i: f"https://collection.sciencemuseumgroup.org.uk/people/cp{i}"
        )
        df_links["QID"] = df_links["QID"].apply(qid_to_url)

        print("adding sameAs relationships for people & orgs")
        add_triples(df_links, OWL.sameAs, subject_col="LINK_ID", object_col="QID")

    else:
        print(
            f"Path {pickle_path} does not exist. No sameAs relationships loaded for people & orgs."
        )


if __name__ == "__main__":

    datastore.create_index()
    load_people_data()
    load_orgs_data()
    load_object_data()
    load_maker_data()
    load_user_data()
    load_sameas_people_orgs("../GITIGNORE_DATA/filtering_people_orgs_result.pkl")
