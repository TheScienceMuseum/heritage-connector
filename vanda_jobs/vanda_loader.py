# Generate content tables
# Run from the root of the repo:
# python3 vanda_jobs/scripts/content-table-generations.py -i objects -j ./GITIGNORE_DATA/elastic_export/objects/all -o ./GITIGNORE_DATA/hc_import/content
# python3 vanda_jobs/scripts/content-table-generations.py -i persons -j ./GITIGNORE_DATA/elastic_export/persons/all -o ./GITIGNORE_DATA/hc_import/content
# python3 vanda_jobs/scripts/content-table-generations.py -i organisations -j ./GITIGNORE_DATA/elastic_export/organisations/all -o ./GITIGNORE_DATA/hc_import/content

# Generate join tables
# Run from the root of repo:
# python3 vanda_jobs/scripts/join-table-generations.py -j ./GITIGNORE_DATA/elastic_export/objects/all -o ./GITIGNORE_DATA/hc_import/join

import sys

sys.path.append("..")

import pandas as pd
import rdflib
import string
import re
import os
from heritageconnector.config import config, field_mapping
from heritageconnector import datastore
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
    collection_name="vanda", field_mapping=field_mapping
)

#  ======================================================

## Content Table Loading

def load_object_data(data_path):
    """Load data from ndjson files """

    table_name = "OBJECT"
    object_df = pd.read_json(data_path, lines=True, nrows=max_records)

    logger.info("loading object data")
    record_loader.add_records(table_name, object_df, add_type=WD.Q488383)

    return

def load_person_data(data_path):
    """Load data from ndjson files """

    table_name = "PERSON"
    person_df = pd.read_json(data_path, lines=True, nrows=max_records)

    logger.info("loading person data")
    record_loader.add_records(table_name, person_df, add_type=WD.Q5)

    return

def load_org_data(data_path):
    """Load data from ndjson files """

    table_name = "ORGANISATION"
    org_df = pd.read_json(data_path, lines=True, nrows=max_records)

    logger.info("loading org data")
    record_loader.add_records(table_name, org_df, add_type=WD.Q43229)

    return

## Join Table Loading

def load_join_data(data_path):
    """Load subject-object-predicate triple from ndjson files and add to existing records"""
    join_df = pd.read_json(data_path, lines=True, nrows=max_records)
    join_df = join_df.rename(columns={"URI_1": "OBJECT", "URI_2": "SUBJECT"})

    logger.info("loading maker data (made by & made)")
    maker_df = join_df[join_df['relationship'] == 'made_by']
    manufactured_df = join_df[join_df['relationship'] == 'manufactured_by']
    made_df = pd.concat([maker_df, manufactured_df])
    # made
    record_loader.add_triples(
        made_df, FOAF.maker, subject_col="SUBJECT", object_col="OBJECT"
    )
    # made by
    record_loader.add_triples(
        made_df, FOAF.made, subject_col="OBJECT", object_col="SUBJECT"
    )

    logger.info("loading depicts data (depicts and depicted)")
    depicts_df = join_df[join_df['relationship'] == 'depicts']
    # depicts - A thing depicted in this representation
    record_loader.add_triples(
        depicts_df, FOAF.depicts, subject_col="OBJECT", object_col="SUBJECT"
    )
    # depiction - A depiction of some thing
    record_loader.add_triples(
        depicts_df, FOAF.depiction, subject_col="SUBJECT", object_col="OBJECT"
    )

    logger.info("loading associated data (significant_person and )")
    associate_df = join_df[join_df['relationship'] == 'associated_with']
    # significant_person - person linked to the item in any possible way
    record_loader.add_triples(
        associate_df, WDT.P3342, subject_col="OBJECT", object_col="SUBJECT"
    )
    # significant_to
    record_loader.add_triples(
        associate_df, WDT.Q67185741, subject_col="SUBJECT", object_col="OBJECT"
    )

    return

if __name__ == "__main__":
    object_data_path = ("../GITIGNORE_DATA/hc_import/content/20210422/objects.ndjson")
    person_data_path = "../GITIGNORE_DATA/hc_import/content/20210422/persons.ndjson"
    org_data_path = "../GITIGNORE_DATA/hc_import/content/20210422/organisations.ndjson"
    join_data_path = "../GITIGNORE_DATA/hc_import/join/20210422/joins.ndjson"

    datastore.create_index()

    load_object_data(object_data_path)
    load_person_data(person_data_path)
    load_org_data(org_data_path)
    load_join_data(join_data_path)
    

    # load_related_from_wikidata()
    # load_sameas_from_wikidata_smg_people_id()
    # load_sameas_people_orgs("../GITIGNORE_DATA/filtering_people_orgs_result.pkl")
    # load_organisation_types("../GITIGNORE_DATA/organisations_with_types.pkl")
    # load_object_types("../GITIGNORE_DATA/objects_with_types.pkl")
    # load_crowdsourced_links(
    #     "../GITIGNORE_DATA/smg-datasets-private/wikidatacapture_plus_kd_links_121120.csv"
    # )
    # load_sameas_from_disambiguator(
    #     "s3://heritageconnector/disambiguation/people_281020/people_preds_positive.csv",
    #     "people",
    # )
    # load_sameas_from_disambiguator(
    #     "s3://heritageconnector/disambiguation/organisations_021120/orgs_preds_positive.csv",
    #     "organisations",
    # )
    # load_sameas_from_disambiguator(
    #     "s3://heritageconnector/disambiguation/objects_131120/test_photographic_aeronautics/preds_positive.csv",
    #     "objects (photographic technology & aeronautics)",
    # )
    # load_sameas_from_disambiguator(
    #     "s3://heritageconnector/disambiguation/objects_131120/test_computing_space/preds_positive.csv",
    #     "objects (computing & space)",
    # )
    # load_sameas_from_disambiguator(
    #     "s3://heritageconnector/disambiguation/objects_131120/test_locomotives_and_rolling_stock/preds_positive.csv",
    #     "objects (locomotives & rolling stock)",
    # )
    # load_ner_annotations(
    #     "en_core_web_lg",
    #     nel_training_data_path="../GITIGNORE_DATA/NEL/review_data_1103.xlsx",
    # )
