import sys

sys.path.append("..")

import pandas as pd
import rdflib
import string
import re
import os
from heritageconnector.config import config, field_mapping
from heritageconnector import datastore, datastore_helpers
from heritageconnector.namespace import (
    XSD,
    FOAF,
    OWL,
    RDF,
    PROV,
    SDO,
    SKOS,
    WD,
    WDT,
)
from heritageconnector.utils.data_transformation import get_year_from_date_value
from heritageconnector.entity_matching.lookup import (
    get_internal_urls_from_wikidata,
    get_sameas_links_from_external_id,
    DenonymConverter,
    get_wikidata_uri_from_placename,
)
from heritageconnector.utils.generic import flatten_list_of_lists
from heritageconnector.utils.wikidata import qid_to_url
from heritageconnector import logging

logger = logging.get_logger(__name__)

# disable pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# optional limit of number of records to import to test loader. no limit -> None
# passed as an argument into `pd.read_csv`. You might want to use your own implementation
# depending on your source data format
max_records = None

# create instance of RecordLoader from datastore
record_loader = datastore.RecordLoader(
    collection_name="SMG", field_mapping=field_mapping
)

#  =============== SMG-specific variables ===============
# these are left in rather than using SMGP/SMGO in heritageconnector.namespace as they serve a slightly
# different purpose: they are meant for converting IDs in internal documents into SMG URLs.
collection_prefix = "https://collection.sciencemuseumgroup.org.uk/objects/co"
people_prefix = "https://collection.sciencemuseumgroup.org.uk/people/cp"

# used for `get_wikidata_uri_from_placename`. Generate your own CSV using the notebook at `experiments/disambiguating place names (geocoding).ipynb`
placename_qid_mapping = pd.read_pickle("s3://heritageconnector/placenames_to_qids.pkl")
#  ======================================================


def load_object_data(catalogue_data_path):
    """Load data from CSV files """

    table_name = "OBJECT"
    catalogue_df = pd.read_csv(catalogue_data_path, low_memory=False, nrows=max_records)
    catalogue_df["URI"] = collection_prefix + catalogue_df["MKEY"].astype(str)
    catalogue_df["MATERIALS"] = catalogue_df["MATERIALS"].apply(
        datastore_helpers.split_list_string
    )
    catalogue_df["ITEM_NAME"] = catalogue_df["ITEM_NAME"].apply(
        datastore_helpers.split_list_string
    )
    catalogue_df["DESCRIPTION"] = catalogue_df["DESCRIPTION"].apply(
        datastore_helpers.process_text
    )
    catalogue_df["DATE_MADE"] = catalogue_df["DATE_MADE"].apply(
        get_year_from_date_value
    )

    logger.info("loading object data")
    record_loader.add_records(table_name, catalogue_df)

    return


def load_people_data(people_data_path):
    """Load data from CSV files """

    # identifier in field_mapping
    table_name = "PERSON"

    people_df = pd.read_csv(people_data_path, low_memory=False, nrows=max_records)
    # TODO: use isIndividual flag here
    people_df = people_df[people_df["GENDER"].isin(["M", "F"])]

    # PREPROCESS
    people_df["URI"] = people_prefix + people_df["LINK_ID"].astype(str)
    # remove punctuation and capitalise first letter
    people_df["TITLE_NAME"] = people_df["TITLE_NAME"].apply(
        lambda i: str(i)
        .capitalize()
        .translate(str.maketrans("", "", string.punctuation))
    )
    people_df["BIRTH_DATE"] = people_df["BIRTH_DATE"].apply(get_year_from_date_value)
    people_df["DEATH_DATE"] = people_df["DEATH_DATE"].apply(get_year_from_date_value)
    people_df["OCCUPATION"] = people_df["OCCUPATION"].apply(
        datastore_helpers.split_list_string
    )
    people_df["NATIONALITY"] = people_df["NATIONALITY"].apply(
        datastore_helpers.split_list_string
    )
    people_df["NATIONALITY"] = people_df["NATIONALITY"].apply(
        lambda x: flatten_list_of_lists(
            [datastore_helpers.get_country_from_nationality(i) for i in x]
        )
    )

    people_df["BIRTH_PLACE"] = people_df["BIRTH_PLACE"].apply(
        lambda i: get_wikidata_uri_from_placename(i, False, placename_qid_mapping)
    )
    people_df["DEATH_PLACE"] = people_df["DEATH_PLACE"].apply(
        lambda i: get_wikidata_uri_from_placename(i, False, placename_qid_mapping)
    )
    people_df[["adlib_id", "adlib_DESCRIPTION", "DESCRIPTION", "NOTE"]] = people_df[
        ["adlib_id", "adlib_DESCRIPTION", "DESCRIPTION", "NOTE"]
    ].fillna("")
    people_df["adlib_id"] = people_df["adlib_id"].apply(
        lambda i: [
            f"https://collection.sciencemuseumgroup.org.uk/people/{x}"
            for x in str(i).split(",")
        ]
        if i
        else ""
    )
    # remove newlines and tab chars
    people_df.loc[:, ["DESCRIPTION", "adlib_DESCRIPTION", "NOTE"]] = people_df.loc[
        :, ["DESCRIPTION", "adlib_DESCRIPTION", "NOTE"]
    ].applymap(datastore_helpers.process_text)

    # create combined text fields
    newline = " \n "  # can't insert into fstring below
    people_df.loc[:, "BIOGRAPHY"] = people_df[
        ["DESCRIPTION", "adlib_DESCRIPTION", "NOTE"]
    ].apply(lambda x: f"{newline.join(x)}" if any(x) else "", axis=1)

    people_df.loc[:, "GENDER"] = people_df.loc[:, "GENDER"].replace(
        {"F": WD.Q6581072, "M": WD.Q6581097}
    )

    logger.info("loading people data")
    record_loader.add_records(table_name, people_df, add_type=WD.Q5)


def load_orgs_data(people_data_path):
    # identifier in field_mapping
    table_name = "ORGANISATION"

    org_df = pd.read_csv(people_data_path, low_memory=False, nrows=max_records)
    # TODO: use isIndividual flag here
    org_df = org_df[org_df["GENDER"] == "N"]

    # PREPROCESS
    org_df["URI"] = people_prefix + org_df["LINK_ID"].astype(str)

    org_df["BIRTH_DATE"] = org_df["BIRTH_DATE"].apply(get_year_from_date_value)
    org_df["DEATH_DATE"] = org_df["DEATH_DATE"].apply(get_year_from_date_value)

    org_df[["adlib_id", "adlib_DESCRIPTION", "DESCRIPTION", "NOTE"]] = org_df[
        ["adlib_id", "adlib_DESCRIPTION", "DESCRIPTION", "NOTE"]
    ].fillna("")
    org_df[["DESCRIPTION", "adlib_DESCRIPTION", "NOTE"]] = org_df[
        ["DESCRIPTION", "adlib_DESCRIPTION", "NOTE"]
    ].applymap(datastore_helpers.process_text)
    org_df[["OCCUPATION", "NATIONALITY"]] = org_df[
        ["OCCUPATION", "NATIONALITY"]
    ].applymap(datastore_helpers.split_list_string)

    org_df["NATIONALITY"] = org_df["NATIONALITY"].apply(
        lambda x: flatten_list_of_lists(
            [datastore_helpers.get_country_from_nationality(i) for i in x]
        )
    )

    org_df["adlib_id"] = org_df["adlib_id"].apply(
        lambda i: [
            f"https://collection.sciencemuseumgroup.org.uk/people/{x}"
            for x in str(i).split(",")
        ]
        if i
        else ""
    )

    newline = " \n "  # can't insert into fstring below
    org_df.loc[:, "BIOGRAPHY"] = org_df[
        ["DESCRIPTION", "adlib_DESCRIPTION", "NOTE"]
    ].apply(lambda x: f"{newline.join(x)}" if any(x) else "", axis=1)

    logger.info("loading orgs data")
    record_loader.add_records(table_name, org_df)

    # also add type organization (Q43229)
    org_df["type_org"] = qid_to_url("Q43229")
    record_loader.add_triples(
        org_df, RDF.type, subject_col="URI", object_col="type_org"
    )

    return


def load_maker_data(maker_data_path, people_data_path):
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
    record_loader.add_triples(
        people_makers, FOAF.maker, subject_col="OBJECT_ID", object_col="PERSON_ORG_ID"
    )

    # where we don't have gender information use FOAF.maker as it's human-readable
    undefined_makers = maker_df[maker_df["GENDER"].isna()]
    record_loader.add_triples(
        undefined_makers,
        FOAF.maker,
        subject_col="OBJECT_ID",
        object_col="PERSON_ORG_ID",
    )

    # use 'product or material produced' Wikidata property for organisations
    orgs_makers = maker_df[maker_df["GENDER"] == "N"]
    record_loader.add_triples(
        orgs_makers, WDT.P1056, subject_col="PERSON_ORG_ID", object_col="OBJECT_ID"
    )

    return


def load_user_data(user_data_path):
    """Load object -> user -> people relationships from CSV files and add to existing records"""
    user_df = pd.read_csv(user_data_path, low_memory=False, nrows=max_records)

    user_df["MKEY"] = collection_prefix + user_df["MKEY"].astype(str)
    user_df["LINK_ID"] = people_prefix + user_df["LINK_ID"].astype(str)
    user_df = user_df.rename(columns={"MKEY": "OBJECT", "LINK_ID": "SUBJECT"})

    logger.info("loading user data (used by & used)")
    # uses
    record_loader.add_triples(
        user_df, WDT.P2283, subject_col="SUBJECT", object_col="OBJECT"
    )
    # used by
    record_loader.add_triples(
        user_df, WDT.P1535, subject_col="OBJECT", object_col="SUBJECT"
    )

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
        record_loader.add_triples(
            df_links, OWL.sameAs, subject_col="LINK_ID", object_col="QID"
        )

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
    record_loader.add_triples(
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

    record_loader.add_triples(
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

    record_loader.add_triples(
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

    record_loader.add_triples(
        df, OWL.sameAs, subject_col="courl", object_col="wikidataurl"
    )


def load_sameas_from_wikidata_smg_people_id():
    logger.info("adding sameAs relationships from Wikidata SMG People ID")

    df = get_sameas_links_from_external_id("P4389")
    df["external_url"] = df["external_url"].str.replace(
        "sciencemuseum.org.uk", "sciencemuseumgroup.org.uk"
    )

    record_loader.add_triples(
        df, OWL.sameAs, subject_col="external_url", object_col="wikidata_url"
    )


def load_sameas_from_disambiguator(path: str, name: str):
    logger.info(f"adding sameAs relationships from {name} disambiguator")

    df = pd.read_csv(path)
    df["wikidata_url"] = df["wikidata_id"].apply(qid_to_url)

    record_loader.add_triples(
        df, OWL.sameAs, subject_col="internal_id", object_col="wikidata_url"
    )


if __name__ == "__main__":
    people_data_path = "../GITIGNORE_DATA/mimsy_adlib_joined_people.csv"
    object_data_path = (
        "../GITIGNORE_DATA/smg-datasets-private/mimsy-catalogue-export.csv"
    )
    maker_data_path = "../GITIGNORE_DATA/smg-datasets-private/items_makers.csv"
    user_data_path = "../GITIGNORE_DATA/smg-datasets-private/items_users.csv"

    datastore.create_index()
    load_people_data(people_data_path)
    load_orgs_data(people_data_path)
    load_object_data(object_data_path)
    load_maker_data(maker_data_path, people_data_path)
    load_user_data(user_data_path)
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
