# Generate content tables
# Run from the root of the repo:
# python3 vanda_jobs/scripts/content-table-generations.py -i objects -j ./GITIGNORE_DATA/elastic_export/objects/custom -g -o ./GITIGNORE_DATA/hc_import/content
# python3 vanda_jobs/scripts/content-table-generations.py -i persons -j ./GITIGNORE_DATA/elastic_export/persons/all -b -o ./GITIGNORE_DATA/hc_import/content
# python3 vanda_jobs/scripts/content-table-generations.py -i organisations -j ./GITIGNORE_DATA/elastic_export/organisations/all -b -o ./GITIGNORE_DATA/hc_import/content
# python3 vanda_jobs/scripts/content-table-generations.py -i events -j ./GITIGNORE_DATA/elastic_export/events/all -g -o ./GITIGNORE_DATA/hc_import/content

# Generate join tables
# Run from the root of repo:
# python3 vanda_jobs/scripts/join-table-generations.py -j ./GITIGNORE_DATA/elastic_export/objects/custom -g -o ./GITIGNORE_DATA/hc_import/join

import sys

sys.path.append("..")

import os
import random
import re
import string

import pandas as pd
import rdflib
from heritageconnector import datastore, datastore_helpers, logging
from heritageconnector.config import config, field_mapping
from heritageconnector.namespace import (FOAF, OWL, PROV, RDF, SDO, SKOS, WD,
                                         WDT, XSD)

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

def reverse_person_preferred_name_and_strip_brackets(name: str) -> str:
    name_stripped = re.sub(r"\([^()]*\)", "", name)

    if not pd.isnull(name_stripped) and len(name_stripped.split(",")) == 2:
        return f"{name_stripped.split(',')[1].strip()} {name_stripped.split(',')[0].strip()}"
    else:
        return name_stripped


def create_object_disambiguating_description(row: pd.Series) -> str:
    """
    Original description col = DESCRIPTION.
    Components:
    - OBJECT_TYPE -> 'Pocket watch.'
    - PLACE_MADE + DATE_MADE -> 'Made in London, 1940.'
    - DESCRIPTION (original description)
    NOTE: must be used before dates are converted to numbers using `get_year_from_date_string`, so that
    uncertainty such as 'about 1971' is added to the description.
    """

    # OBJECT_TYPE
    # Here we also check that the name without 's' or 'es' is not already in the description,
    # which should cover the majority of plurals.
    if (
        (str(row.OBJECT_TYPE) != "nan")
        and (str(row.OBJECT_TYPE).lower() not in row.DESCRIPTION.lower())
        and (str(row.OBJECT_TYPE).rstrip("s").lower() not in row.DESCRIPTION.lower())
        and (str(row.OBJECT_TYPE).rstrip("es").lower() not in row.DESCRIPTION.lower())
    ):
        object_type = f"{row.OBJECT_TYPE.capitalize().strip()}."
    else:
        object_type = ""

    # PRIMARY_PLACE + PRIMARY_DATE
    add_place_made = (row["PRIMARY_PLACE"]) and (str(row["PRIMARY_PLACE"]) != "nan") and (
        str(row["PRIMARY_PLACE"]).lower() not in row.DESCRIPTION.lower()
    )

    add_date_made = (row["PRIMARY_DATE"]) and (str(row["PRIMARY_DATE"]) != "nan") and (str(row["PRIMARY_DATE"])) and (
        str(row["PRIMARY_DATE"]) not in row.DESCRIPTION.lower()
    )
    # Also check for dates minus suffixes, e.g. 200-250 should match with 200-250 AD and vice-versa
    if re.findall(r"\d+-?\d*", str(row["PRIMARY_DATE"])):
        add_date_made = add_date_made and (
            re.findall(r"\d+-?\d*", row["PRIMARY_DATE"])[0].lower()
            not in row.DESCRIPTION.lower()
        )

    if add_place_made and add_date_made:
        made_str = f"Made in {row.PRIMARY_PLACE.strip()}, {row.PRIMARY_DATE}."
    elif add_place_made:
        made_str = f"Made in {row.PRIMARY_PLACE.strip()}."
    elif add_date_made:
        made_str = f"Made {row.PRIMARY_DATE}."
    else:
        made_str = ""

    # add space and full stop (if needed) to end of description
    if row.DESCRIPTION.strip():
        description = (
            row.DESCRIPTION.strip()
            if row.DESCRIPTION.strip()[-1] == "."
            else f"{row.DESCRIPTION.strip()}."
        )
    else:
        description = ""

    # we shuffle the components of the description so any model using them does not learn the order that we put them in
    aug_description_components = [object_type, description, made_str]
    random.shuffle(aug_description_components)

    return (" ".join(aug_description_components)).strip()

def load_object_data(data_path):
    """Load data from ndjson files """

    table_name = "OBJECT"
    object_df = pd.read_json(data_path, lines=True, nrows=max_records)

    # PRE_PROCESS
    object_df[["DESCRIPTION", "PHYS_DESCRIPTION", "PRODUCTION_TYPE"]] = object_df[["DESCRIPTION", "PHYS_DESCRIPTION", "PRODUCTION_TYPE"]].fillna("")

    # remove newlines and tab chars
    object_df.loc[:, ["DESCRIPTION", "PHYS_DESCRIPTION", "PRODUCTION_TYPE"]] = object_df.loc[
        :, ["DESCRIPTION", "PHYS_DESCRIPTION", "PRODUCTION_TYPE"]
    ].applymap(datastore_helpers.process_text)

    # create combined text fields
    newline = " \n "  # can't insert into fstring below
    object_df.loc[:, "COMBINED_DESCRIPTION"] = object_df[["DESCRIPTION", "PHYS_DESCRIPTION", "PRODUCTION_TYPE"]].apply(
        lambda x: f"{newline.join(x)}" if any(x) else "", axis=1
    )

    object_df["DISAMBIGUATING_DESCRIPTION"] = object_df.apply(
        create_object_disambiguating_description, axis=1
    )

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

def load_event_data(data_path):
    """Load data from ndjson files """

    table_name = "EVENT"
    event_df = pd.read_json(data_path, lines=True, nrows=max_records)

    logger.info("loading event data")
    record_loader.add_records(table_name, event_df, add_type=WD.P793)

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
        made_df, predicate=FOAF.maker, subject_col="OBJECT", object_col="SUBJECT"
    )
    # made by
    record_loader.add_triples(
        made_df, predicate=FOAF.made, subject_col="SUBJECT", object_col="OBJECT"
    )

    logger.info("loading depicts data (depicts and depicted)")
    depicts_df = join_df[join_df['relationship'] == 'depicts']
    # depicts - A thing depicted in this representation
    record_loader.add_triples(
        depicts_df, predicate=FOAF.depicts, subject_col="OBJECT", object_col="SUBJECT"
    )
    # depiction - A depiction of some thing
    record_loader.add_triples(
        depicts_df, predicate=FOAF.depiction, subject_col="SUBJECT", object_col="OBJECT"
    )

    logger.info("loading associated data (significant_person and )")
    associate_df = join_df[join_df['relationship'] == 'associated_with']
    # significant_person - person linked to the item in any possible way
    record_loader.add_triples(
        associate_df, predicate=WDT.P3342, subject_col="OBJECT", object_col="SUBJECT"
    )
    # significant_to
    record_loader.add_triples(
        associate_df,  predicate=WDT.Q67185741, subject_col="SUBJECT", object_col="OBJECT"
    )

    logger.info("loading materials data (made from material and uses_this_material)")
    materials_df = join_df[join_df['relationship'] == 'made_from_material']
    # made_from_material
    record_loader.add_triples(
        materials_df, predicate=WDT.P186, subject_col="OBJECT", object_col="SUBJECT"
    )
    # uses_this_material
    record_loader.add_triples(
        materials_df, predicate=WDT.Q104626285, subject_col="SUBJECT", object_col="OBJECT"
    )

    logger.info("loading techniques data (fabrication_method)")
    technique_df = join_df[join_df['relationship'] == 'fabrication_method']
    # fabrication_method
    record_loader.add_triples(
        technique_df, predicate=WDT.P2079, subject_col="OBJECT", object_col="SUBJECT"
    )

    record_loader.add_triples(
        technique_df, predicate=WDT.P2079, subject_col="SUBJECT", object_col="OBJECT"
    )

    logger.info("loading events data (significant_event)")
    events_df = join_df[join_df['relationship'] == 'significant_event']
    # significant event
    record_loader.add_triples(
        events_df, predicate=WDT.P793, subject_col="OBJECT", object_col="SUBJECT"
    )

    record_loader.add_triples(
        events_df, predicate=WDT.P793, subject_col="SUBJECT", object_col="OBJECT"
    )

    return

if __name__ == "__main__":
    object_data_path = ("../GITIGNORE_DATA/hc_import/content/20210705/objects.ndjson")
    person_data_path = "../GITIGNORE_DATA/hc_import/content/20210705/persons.ndjson"
    org_data_path = "../GITIGNORE_DATA/hc_import/content/20210705/organisations.ndjson"
    event_data_path = "../GITIGNORE_DATA/hc_import/content/20210705/events.ndjson"
    join_data_path = "../GITIGNORE_DATA/hc_import/join/20210705/joins.ndjson"

    datastore.create_index()

    load_object_data(object_data_path)
    load_person_data(person_data_path)
    load_org_data(org_data_path)
    load_event_data(event_data_path)
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
