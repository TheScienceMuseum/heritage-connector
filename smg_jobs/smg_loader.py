import sys

sys.path.append("..")

import pandas as pd
import rdflib
from rdflib import Graph, Literal, URIRef
from rdflib.serializer import Serializer
import json
import string
import re
import os
from tqdm.auto import tqdm
from heritageconnector.config import config, field_mapping
from heritageconnector import datastore
from heritageconnector.namespace import XSD, FOAF, OWL, RDF, PROV, SDO, SKOS, WD, WDT
from heritageconnector.utils.data_transformation import get_year_from_date_value
from heritageconnector.entity_matching.lookup import (
    get_internal_urls_from_wikidata,
    get_sameas_links_from_external_id,
    DenonymConverter,
)
from heritageconnector.utils.generic import flatten_list_of_lists
from heritageconnector.utils.wikidata import qid_to_url
from heritageconnector import logging

logger = logging.get_logger(__name__)

# disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

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
    {"@sdo": "https://schema.org/", "@language": "en"},
    {"@owl": "http://www.w3.org/2002/07/owl#", "@language": "en"},
    {"@xsd": "http://www.w3.org/2001/XMLSchema#", "@language": "en"},
    {"@wd": "http://www.wikidata.org/entity/", "@language": "en"},
    {"@wdt": "http://www.wikidata.org/prop/direct/", "@language": "en"},
    {"@prov": "http://www.w3.org/ns/prov#", "@language": "en"},
    {"@rdfs": "http://www.w3.org/2000/01/rdf-schema#", "@language": "en"},
    {"@skos": "http://www.w3.org/2004/02/skos/core#", "@language": "en"},
]

collection_prefix = "https://collection.sciencemuseumgroup.org.uk/objects/co"
people_prefix = "https://collection.sciencemuseumgroup.org.uk/people/cp"

# PIDs from field_mapping to store in ES separate to the graph object
non_graph_pids = [
    "description",
    # NOTE: enable the next two lines for KG embedding training (exclude first & last names)
    # WDT.P735, # first name
    # WDT.P734, # last name
]

denonym_converter = DenonymConverter()

# columns of interest are 'place name', 'qid', 'country qid'
placename_qid_mapping = pd.read_pickle("s3://heritageconnector/placenames_to_qids.pkl")


def get_wiki_uri_from_placename(place_name: str, get_country: bool) -> rdflib.URIRef:
    """
    Get URI of QID from place name. `get_country` flag returns the QID of the country instead of the place.
    """

    if str(place_name).lower() not in placename_qid_mapping["place name"].tolist():
        return None

    if get_country:
        return_uri = placename_qid_mapping.loc[
            placename_qid_mapping["place name"] == str(place_name).lower(),
            "country_qid",
        ].values[0]
    else:
        return_uri = placename_qid_mapping.loc[
            placename_qid_mapping["place name"] == str(place_name).lower(), "qid"
        ].values[0]

    if str(return_uri) == "nan":
        return None
    else:
        return URIRef(return_uri)


def get_country_from_nationality(nationality):
    country = denonym_converter.get_country_from_nationality(nationality)

    if country is not None:
        return country
    else:
        return nationality


def process_text(text: str):
    """
    Remove newlines/other problematic characters
    """
    newstr = str(text)
    newstr = newstr.replace("\n", " ")
    newstr = newstr.replace("\t", " ")

    return newstr


def split_list_string(l: list):
    """
    Splits string separated by either commas or semicolons into a lowercase list.
    """

    return [
        x.strip().lower()
        for x in str(l).replace(";", ",").split(",")
        if x.strip() != ""
    ]


def load_object_data():
    """Load data from CSV files """

    table_name = "OBJECT"
    catalogue_df = pd.read_csv(catalogue_data_path, low_memory=False, nrows=max_records)
    catalogue_df = catalogue_df.rename(columns={"MKEY": "ID"})
    catalogue_df["PREFIX"] = collection_prefix
    catalogue_df["MATERIALS"] = catalogue_df["MATERIALS"].apply(split_list_string)
    catalogue_df["ITEM_NAME"] = catalogue_df["ITEM_NAME"].apply(split_list_string)
    catalogue_df["DESCRIPTION"] = catalogue_df["DESCRIPTION"].apply(process_text)
    catalogue_df["DATE_MADE"] = catalogue_df["DATE_MADE"].apply(
        get_year_from_date_value
    )

    logger.info("loading object data")
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
    # remove punctuation and capitalise first letter
    people_df["TITLE_NAME"] = people_df["TITLE_NAME"].apply(
        lambda i: str(i)
        .capitalize()
        .translate(str.maketrans("", "", string.punctuation))
    )
    people_df["BIRTH_DATE"] = people_df["BIRTH_DATE"].apply(get_year_from_date_value)
    people_df["DEATH_DATE"] = people_df["DEATH_DATE"].apply(get_year_from_date_value)
    people_df["OCCUPATION"] = people_df["OCCUPATION"].apply(split_list_string)
    people_df["NATIONALITY"] = people_df["NATIONALITY"].apply(split_list_string)
    people_df["NATIONALITY"] = people_df["NATIONALITY"].apply(
        lambda x: flatten_list_of_lists([get_country_from_nationality(i) for i in x])
    )

    people_df["BIRTH_PLACE"] = people_df["BIRTH_PLACE"].apply(
        lambda i: get_wiki_uri_from_placename(i, False)
    )
    people_df["DEATH_PLACE"] = people_df["DEATH_PLACE"].apply(
        lambda i: get_wiki_uri_from_placename(i, False)
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
    people_df.loc[:, "GENDER"] = people_df.loc[:, "GENDER"].replace(
        {"F": WD.Q6581072, "M": WD.Q6581097}
    )

    logger.info("loading people data")
    add_records(table_name, people_df, add_type=WD.Q5)


def load_orgs_data():
    # identifier in field_mapping
    table_name = "ORGANISATION"

    org_df = pd.read_csv(people_data_path, low_memory=False, nrows=max_records)
    # TODO: use isIndividual flag here
    org_df = org_df[org_df["GENDER"] == "N"]

    # PREPROCESS
    org_df = org_df.rename(columns={"LINK_ID": "ID"})
    org_df["PREFIX"] = people_prefix

    org_df["BIRTH_DATE"] = org_df["BIRTH_DATE"].apply(get_year_from_date_value)
    org_df["DEATH_DATE"] = org_df["DEATH_DATE"].apply(get_year_from_date_value)

    org_df["DESCRIPTION"] = org_df["DESCRIPTION"].apply(process_text)
    org_df["BRIEF_BIO"] = org_df["BRIEF_BIO"].apply(process_text)
    org_df["OCCUPATION"] = org_df["OCCUPATION"].apply(split_list_string)
    org_df["NATIONALITY"] = org_df["NATIONALITY"].apply(split_list_string)
    org_df["NATIONALITY"] = org_df["NATIONALITY"].apply(
        lambda x: flatten_list_of_lists([get_country_from_nationality(i) for i in x])
    )
    # org_df["NATIONALITY"] = org_df["NATIONALITY"].apply(lambda x: [get_wiki_uri_from_placename(i, True) for i in x])

    logger.info("loading orgs data")
    add_records(table_name, org_df)

    # also add type organization (Q43229)
    org_df["URI"] = org_df["ID"].apply(lambda i: people_prefix + str(i))
    org_df["type_org"] = qid_to_url("Q43229")
    add_triples(org_df, RDF.type, subject_col="URI", object_col="type_org")

    return


def load_maker_data():
    """Load object -> maker -> people relationships from CSV files and add to existing records """
    # import people_orgs so we can split maker_df into people and organisations using the gender column
    #
    maker_df = pd.read_csv(maker_data_path, low_memory=False, nrows=max_records)
    people_orgs_df = pd.read_csv(people_data_path, low_memory=False, nrows=max_records)

    maker_df = maker_df.merge(people_orgs_df[["LINK_ID", "GENDER"]], how="left")
    maker_df["MKEY"] = collection_prefix + maker_df["MKEY"].astype(str)
    maker_df["LINK_ID"] = people_prefix + maker_df["LINK_ID"].astype(str)
    maker_df = maker_df.rename(
        columns={"MKEY": "OBJECT_ID", "LINK_ID": "PERSON_ORG_ID"}
    )

    logger.info("loading maker data for people and orgs")
    people_makers = maker_df[maker_df["GENDER"].isin(["M", "F"])]
    add_triples(
        people_makers, FOAF.maker, subject_col="OBJECT_ID", object_col="PERSON_ORG_ID"
    )

    # where we don't have gender information use FOAF.maker as it's human-readable
    undefined_makers = maker_df[maker_df["GENDER"].isna()]
    add_triples(
        undefined_makers,
        FOAF.maker,
        subject_col="OBJECT_ID",
        object_col="PERSON_ORG_ID",
    )

    # use 'product or material produced' Wikidata property for organisations
    orgs_makers = maker_df[maker_df["GENDER"] == "N"]
    add_triples(
        orgs_makers, WDT.P1056, subject_col="PERSON_ORG_ID", object_col="OBJECT_ID"
    )

    return


def load_user_data():
    """Load object -> user -> people relationships from CSV files and add to existing records"""
    user_df = pd.read_csv(user_data_path, low_memory=False, nrows=max_records)

    user_df["MKEY"] = collection_prefix + user_df["MKEY"].astype(str)
    user_df["LINK_ID"] = people_prefix + user_df["LINK_ID"].astype(str)
    user_df = user_df.rename(columns={"MKEY": "OBJECT", "LINK_ID": "SUBJECT"})

    logger.info("loading user data (used by & used)")
    # uses
    add_triples(user_df, WDT.P2283, subject_col="SUBJECT", object_col="OBJECT")
    # used by
    add_triples(user_df, WDT.P1535, subject_col="OBJECT", object_col="SUBJECT")

    return


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

        logger.info("adding sameAs relationships for people & orgs")
        add_triples(df_links, OWL.sameAs, subject_col="LINK_ID", object_col="QID")

    else:
        logger.warn(
            f"Path {pickle_path} does not exist. No sameAs relationships loaded for people & orgs."
        )


def load_related_from_wikidata():
    """Load relatedMatch connections that already exist between the SMG records and Wikidata"""
    logger.info("adding relatedMatch relationships from Wikidata URLs")

    connection_df = get_internal_urls_from_wikidata(
        "collection.sciencemuseum.org.uk", config.WIKIDATA_SPARQL_ENDPOINT
    )

    # remove anything after c(o|d|p)(\d+)
    connection_df["internalURL"] = connection_df["internalURL"].apply(
        lambda x: re.findall(r"https://(?:\w.+)/c(?:o|d|p)(?:\d+)", x)[0]
    )
    connection_df["internalURL"] = connection_df["internalURL"].str.replace(
        "sciencemuseum.org.uk", "sciencemuseumgroup.org.uk"
    )
    add_triples(
        connection_df, SKOS.relatedMatch, subject_col="internalURL", object_col="item"
    )


def load_organisation_types(org_type_df_path):
    logger.info("adding resolved organisation types")

    org_type_df = pd.read_pickle(org_type_df_path)
    org_type_df = org_type_df.loc[
        org_type_df["OCCUPATION_resolved"].apply(len) > 0,
        ["LINK_ID", "OCCUPATION_resolved"],
    ]

    org_type_df["LINK_ID"] = org_type_df["LINK_ID"].astype(str)
    org_type_df["ID"] = people_prefix + org_type_df["LINK_ID"]
    org_type_df["OCCUPATION_resolved"] = org_type_df["OCCUPATION_resolved"].apply(
        qid_to_url
    )

    add_triples(
        org_type_df, RDF.type, subject_col="ID", object_col="OCCUPATION_resolved"
    )


def load_object_types(object_type_df_path):
    logger.info("adding resolved object types")

    object_type_df = pd.read_pickle(object_type_df_path)
    object_type_df = object_type_df.loc[
        object_type_df["ITEM_NAME_resolved"].apply(len) > 0,
        ["MKEY", "ITEM_NAME_resolved"],
    ]

    object_type_df["MKEY"] = object_type_df["MKEY"].astype(str)
    object_type_df["ID"] = collection_prefix + object_type_df["MKEY"]
    object_type_df["ITEM_NAME_resolved"] = object_type_df["ITEM_NAME_resolved"].apply(
        qid_to_url
    )

    add_triples(
        object_type_df, RDF.type, subject_col="ID", object_col="ITEM_NAME_resolved"
    )


def load_crowdsourced_links(links_path):
    logger.info("adding crowdsourced links")

    df = pd.read_csv(links_path)
    df["courl"] = df["courl"].str.replace(
        "http://collectionsonline-staging.eu-west-1.elasticbeanstalk.com/",
        "https://collection.sciencemuseumgroup.org.uk/",
    )
    df["courl"] = df["courl"].str.replace(
        "http://localhost:8000/", "https://collection.sciencemuseumgroup.org.uk/"
    )
    # remove anything after c(o|d|p)(\d+)
    df["courl"] = df["courl"].apply(
        lambda x: re.findall(r"https://(?:\w.+)/(?:co|cp|ap)(?:\d+)", x)[0]
    )
    df["wikidataurl"] = df["wikidataurl"].str.replace("https", "http")
    df["wikidataurl"] = df["wikidataurl"].str.replace("/wiki/", "/entity/")

    add_triples(df, OWL.sameAs, subject_col="courl", object_col="wikidataurl")


def load_sameas_from_wikidata_smg_people_id():
    logger.info("adding sameAs relationships from Wikidata SMG People ID")

    df = get_sameas_links_from_external_id("P4389")
    df["external_url"] = df["external_url"].str.replace(
        "sciencemuseum.org.uk", "sciencemuseumgroup.org.uk"
    )

    add_triples(df, OWL.sameAs, subject_col="external_url", object_col="wikidata_url")


def load_sameas_from_disambiguator(path: str, name: str):
    logger.info(f"adding sameAs relationships from {name} disambiguator")

    df = pd.read_csv(path)
    df["wikidata_url"] = df["wikidata_id"].apply(qid_to_url)

    add_triples(df, OWL.sameAs, subject_col="internal_id", object_col="wikidata_url")


#  =============== GENERIC FUNCTIONS FOR LOADING (move these?) ===============


def add_record(table_name, row, add_type=False):
    """Create and store new HC record with a JSON-LD graph"""

    uri_prefix = row["PREFIX"]
    uri = uri_prefix + str(row["ID"])

    table_mapping = field_mapping.mapping[table_name]
    data_fields = [
        k for k, v in table_mapping.items() if v.get("PID") in non_graph_pids
    ]

    data = serialize_to_json(table_name, row, data_fields)
    data["uri"] = uri
    jsonld = serialize_to_jsonld(
        table_name, uri, row, ignore_types=non_graph_pids, add_type=add_type
    )

    datastore.create(collection, table_name, data, jsonld)


def add_records(table_name, df, add_type=False):
    """Use ES parallel_bulk mechanism to add records from a table"""
    generator = record_create_generator(table_name, df, add_type)
    datastore.es_bulk(generator, len(df))


def record_create_generator(table_name, df, add_type):
    """Yields jsonld for a row for use with ES bulk helpers"""

    table_mapping = field_mapping.mapping[table_name]

    data_fields = [
        k for k, v in table_mapping.items() if v.get("PID") in non_graph_pids
    ]

    for _, row in df.iterrows():
        uri_prefix = row["PREFIX"]
        uri = uri_prefix + str(row["ID"])

        data = serialize_to_json(table_name, row, data_fields)
        jsonld = serialize_to_jsonld(
            table_name, uri, row, ignore_types=non_graph_pids, add_type=add_type
        )

        doc = {
            "_id": uri,
            "uri": uri,
            "collection": collection,
            "type": table_name,
            "data": data,
            "graph": jsonld,
        }

        yield doc


def add_triples(df, predicate, subject_col="SUBJECT", object_col="OBJECT"):
    """
    Add triples with RDF predicate and dataframe containing subject and object columns.
    Values in object_col can either be string or list. If list, a one subject to many 
    objects relationship is assumed.
    """

    generator = record_update_generator(df, predicate, subject_col, object_col)
    datastore.es_bulk(generator, len(df))


def record_update_generator(df, predicate, subject_col="SUBJECT", object_col="OBJECT"):
    """Yields jsonld docs to update existing records with new triples"""

    for _, row in df.iterrows():
        g = Graph()
        if isinstance(row[object_col], str):
            g.add((URIRef(row[subject_col]), predicate, URIRef(row[object_col])))
        elif isinstance(row[object_col], list):
            [
                g.add((URIRef(row[subject_col]), predicate, URIRef(v)))
                for v in row[object_col]
            ]

        jsonld_dict = json.loads(
            g.serialize(format="json-ld", context=context, indent=4)
        )
        _ = jsonld_dict.pop("@id")
        _ = jsonld_dict.pop("@context")

        body = {"graph": jsonld_dict}

        doc = {"_id": row[subject_col], "_op_type": "update", "doc": body}

        yield doc


def serialize_to_json(table_name: str, row: pd.Series, columns: list) -> dict:
    """Return a JSON representation of data fields to exist outside of the graph."""

    table_mapping = field_mapping.mapping[table_name]

    data = {}

    for col in columns:
        if (
            "RDF" in table_mapping[col]
            and bool(row[col])
            and (str(row[col]).lower() != "nan")
        ):
            # TODO: these lines load description in as https://collection.sciencemuseumgroup.org.uk/objects/co__#<field_name> but for some reason they cause an Elasticsearch timeout
            # key = row['PREFIX'] + str(row['ID']) + "#" + col.lower()
            # data[key] = row[col]
            data.update({table_mapping[col]["RDF"]: row[col]})

    return data


def serialize_to_jsonld(
    table_name: str,
    uri: str,
    row: pd.Series,
    ignore_types: list,
    add_type: rdflib.term.URIRef = False,
) -> dict:
    """
    Returns a JSON-LD represention of a record

    Args:
        table_name (str): given name of the table being imported
        uri (str): URI of subject
        row (pd.Series): DataFrame row (record) to serialize
        ignore_types (list): PIDs to ignore when importing
        add_type (rdflib.term.URIRef, optional): whether to add @type field with the table_name. If a value rather than
            a boolean is passed in, this will be added as the type for the table. Defaults to True.

    Raises:
        KeyError: [description]

    Returns:
        dict: [description]
    """

    g = Graph()
    record = URIRef(uri)

    g.add((record, SKOS.hasTopConcept, Literal(table_name)))

    # Add RDF:type
    # Need to check for isinstance otherwise this will fail silently during bulk load, causing the entire record to not load
    if add_type and isinstance(add_type, rdflib.term.URIRef):
        g.add((record, RDF.type, add_type))

    # This code is effectivly the mapping from source data to the data we care about
    table_mapping = field_mapping.mapping[table_name]

    keys = {
        k
        for k, v in table_mapping.items()
        if k not in ["ID", "PREFIX"] and "RDF" in v and v.get("PID") not in ignore_types
    }

    for col in keys:
        # this will trigger for the first row in the dataframe
        if col not in row.index:
            raise KeyError(f"column {col} not in data for table {table_name}")

        if bool(row[col]) and (str(row[col]).lower() != "nan"):
            if isinstance(row[col], list):
                [
                    g.add((record, table_mapping[col]["RDF"], Literal(val)))
                    for val in row[col]
                    if str(val) != "nan"
                ]
            elif isinstance(row[col], URIRef):
                g.add((record, table_mapping[col]["RDF"], row[col]))

            else:
                g.add((record, table_mapping[col]["RDF"], Literal(row[col])))

    json_ld_dict = json.loads(
        g.serialize(format="json-ld", context=context, indent=4).decode("utf-8")
    )

    # "'@graph': []" appears when there are no linked objects to the document, which breaks the RDF conversion.
    # There is also no @id field in the graph when this happens.
    json_ld_dict.pop("@graph", None)
    json_ld_dict["@id"] = uri

    return json_ld_dict


if __name__ == "__main__":

    datastore.create_index()
    load_people_data()
    load_orgs_data()
    load_object_data()
    load_maker_data()
    load_user_data()
    load_related_from_wikidata()
    load_sameas_from_wikidata_smg_people_id()
    load_sameas_people_orgs("../GITIGNORE_DATA/filtering_people_orgs_result.pkl")
    load_organisation_types("../GITIGNORE_DATA/organisations_with_types.pkl")
    load_object_types("../GITIGNORE_DATA/objects_with_types.pkl")
    load_crowdsourced_links(
        "../GITIGNORE_DATA/smg-datasets-private/wikidatacapture_plus_kd_links_121120.csv"
    )
    load_sameas_from_disambiguator(
        "s3://heritageconnector/disambiguation/people_281020/people_preds_positive.csv",
        "people",
    )
    load_sameas_from_disambiguator(
        "s3://heritageconnector/disambiguation/organisations_021120/orgs_preds_positive.csv",
        "organisations",
    )
    load_sameas_from_disambiguator(
        "s3://heritageconnector/disambiguation/objects_131120/test_photographic_aeronautics/preds_positive.csv",
        "objects (photographic technology & aeronautics)",
    )
    load_sameas_from_disambiguator(
        "s3://heritageconnector/disambiguation/objects_131120/test_computing_space/preds_positive.csv",
        "objects (computing & space)",
    )
    load_sameas_from_disambiguator(
        "s3://heritageconnector/disambiguation/objects_131120/test_locomotives_and_rolling_stock/preds_positive.csv",
        "objects (locomotives & rolling stock)",
    )
