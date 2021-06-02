import sys

sys.path.append("..")

import pandas as pd
import random
import string
import re
import os
from typing import Optional
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
    get_wikidata_uri_from_placename,
)
from heritageconnector.utils.generic import flatten_list_of_lists, get_timestamp
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
document_prefix = "https://collection.sciencemuseumgroup.org.uk/documents/"
adlib_people_prefix = "https://collection.sciencemuseumgroup.org.uk/people/"

# used for `get_wikidata_uri_from_placename`. Generate your own CSV using the notebook at `experiments/disambiguating place names (geocoding).ipynb`
placename_qid_mapping = pd.read_pickle("s3://heritageconnector/placenames_to_qids.pkl")
adlib_placename_qid_mapping = pd.read_csv(
    "s3://heritageconnector/adlib_placenames_to_qids.csv"
)
#  ======================================================


def create_object_disambiguating_description(row: pd.Series) -> str:
    """
    Original description col = DESCRIPTION.
    Components:
    - ITEM_NAME -> 'Pocket watch.'
    - PLACE_MADE + DATE_MADE -> 'Made in London, 1940.'
    - DESCRIPTION (original description)
    NOTE: must be used before dates are converted to numbers using `get_year_from_date_string`, so that
    uncertainty such as 'about 1971' is added to the description.
    """

    # ITEM_NAME
    # Here we also check that the name without 's' or 'es' is not already in the description,
    # which should cover the majority of plurals.
    if (
        (str(row.ITEM_NAME[0]) != "nan")
        and (str(row.ITEM_NAME[0]).lower() not in row.DESCRIPTION.lower())
        and (str(row.ITEM_NAME[0]).rstrip("s").lower() not in row.DESCRIPTION.lower())
        and (str(row.ITEM_NAME[0]).rstrip("es").lower() not in row.DESCRIPTION.lower())
    ):
        item_name = f"{row.ITEM_NAME[0].capitalize().strip()}."
    else:
        item_name = ""

    # PLACE_MADE + DATE_MADE
    add_place_made = (str(row["PLACE_MADE"]) != "nan") and (
        str(row["PLACE_MADE"]).lower() not in row.DESCRIPTION.lower()
    )
    add_date_made = (str(row["DATE_MADE"]) != "nan") and (
        str(row["DATE_MADE"]).lower() not in row.DESCRIPTION.lower()
    )
    # Also check for dates minus suffixes, e.g. 200-250 should match with 200-250 AD and vice-versa
    if re.findall(r"\d+-?\d*", str(row["DATE_MADE"])):
        add_date_made = add_date_made and (
            re.findall(r"\d+-?\d*", row["DATE_MADE"])[0].lower()
            not in row.DESCRIPTION.lower()
        )

    if add_place_made and add_date_made:
        made_str = f"Made in {row.PLACE_MADE.strip()}, {row.DATE_MADE.strip()}."
    elif add_place_made:
        made_str = f"Made in {row.PLACE_MADE.strip()}."
    elif add_date_made:
        made_str = f"Made {row.DATE_MADE.strip()}."
    else:
        made_str = ""

    # add space and full stop (if needed) to end of description
    description = (
        row.DESCRIPTION.strip()
        if row.DESCRIPTION.strip()[-1] == "."
        else f"{row.DESCRIPTION.strip()}."
    )

    # we shuffle the components of the description so any model using them does not learn the order that we put them in
    aug_description_components = [item_name, description, made_str]
    random.shuffle(aug_description_components)

    return (" ".join(aug_description_components)).strip()


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
    catalogue_df.loc[:, ["DESCRIPTION", "OPTION1"]] = catalogue_df.loc[
        :, ["DESCRIPTION", "OPTION1"]
    ].applymap(datastore_helpers.process_text)

    newline = " \n "
    catalogue_df.loc[:, "DESCRIPTION"] = catalogue_df[["DESCRIPTION", "OPTION1"]].apply(
        lambda x: f"{newline.join(x)}"
        if x["DESCRIPTION"] != x["OPTION1"] and (str(x["OPTION1"]) != "nan")
        else x["DESCRIPTION"],
        axis=1,
    )

    catalogue_df["DISAMBIGUATING_DESCRIPTION"] = catalogue_df.apply(
        create_object_disambiguating_description, axis=1
    )

    catalogue_df["DATE_MADE"] = catalogue_df["DATE_MADE"].apply(
        get_year_from_date_value
    )
    catalogue_df = catalogue_df[~catalogue_df["CATEGORY1"].str.contains("Disposal")]
    catalogue_df["CATEGORY1"] = catalogue_df["CATEGORY1"].apply(
        lambda x: x.split(" - ")[1].strip()
    )

    logger.info("loading object data")
    record_loader.add_records(table_name, catalogue_df)

    return


def load_adlib_document_data(adlib_document_data_path):
    document_df = pd.read_csv(
        adlib_document_data_path, low_memory=False, nrows=max_records
    )
    table_name = "DOCUMENT"

    # PREPROCESS
    document_df = document_df.rename(columns={"admin.uid": "ID"})
    document_df = document_df.rename(columns={"summary_title": "TITLE"})
    document_df = document_df.rename(
        columns={"content.description.0.value": "DESCRIPTION"}
    )
    # We won't add any more context to documents here as we are not planning to link named entities
    # to documents. This description will just be used to link entities found within it (which are not
    # the document itself) to people, orgs, or objects.
    document_df["DISAMBIGUATING_DESCRIPTION"] = document_df["DESCRIPTION"].copy()
    document_df = document_df.rename(
        columns={"lifecycle.creation.0.date.0.note.0.value": "DATE_MADE"}
    )
    document_df["URI"] = document_prefix + document_df["ID"].astype(str)

    # SUBJECT (e.g. photography)
    document_df["SUBJECT"] = ""
    subject_cols = [
        col for col in document_df.columns if col.startswith("content.subjects")
    ]
    for idx, row in document_df.iterrows():
        document_df.at[idx, "SUBJECT"] = [
            item for item in row[subject_cols].tolist() if str(item) != "nan"
        ]

    # fonds, maker, agents, web/urls, date-range, measurements, materials?
    document_df["PREFIX"] = document_prefix
    document_df["DESCRIPTION"] = document_df["DESCRIPTION"].apply(
        datastore_helpers.process_text
    )
    document_df["DATE_MADE"] = document_df["DATE_MADE"].apply(get_year_from_date_value)

    logger.info("loading adlib document data")
    record_loader.add_records(table_name, document_df)

    # makers / users / agents
    document_df[
        "lifecycle.creation.0.maker.0.admin.uid"
    ] = adlib_people_prefix + document_df[
        "lifecycle.creation.0.maker.0.admin.uid"
    ].astype(
        str
    )
    record_loader.add_triples(
        document_df,
        FOAF.made,
        subject_col="lifecycle.creation.0.maker.0.admin.uid",
        object_col="URI",
    )
    record_loader.add_triples(
        document_df,
        FOAF.maker,
        subject_col="URI",
        object_col="lifecycle.creation.0.maker.0.admin.uid",
    )

    # when adding prov.used triples, we first need to filter the document dataframe to a new one, which only contains valid URIs in both columns
    # (add triples can't handle empty values)
    related_people_docs_df = document_df[
        pd.notnull(document_df["content.agents.0.admin.uid"])
    ]
    related_people_docs_df[
        "content.agents.0.admin.uid"
    ] = adlib_people_prefix + related_people_docs_df[
        "content.agents.0.admin.uid"
    ].astype(
        str
    )
    record_loader.add_triples(
        related_people_docs_df,
        SKOS.related,
        subject_col="content.agents.0.admin.uid",
        object_col="URI",
    )
    record_loader.add_triples(
        related_people_docs_df,
        SKOS.related,
        subject_col="URI",
        object_col="content.agents.0.admin.uid",
    )


def load_adlib_people_data(adlib_people_data_path):
    table_name = "PERSON_ADLIB"

    people_df = pd.read_csv(adlib_people_data_path, low_memory=False, nrows=max_records)

    # PREPROCESS
    people_df = people_df[people_df["type.type"] == "person"]
    people_df = people_df.rename(columns={"admin.uid": "ID"})
    people_df = people_df.rename(columns={"name.0.title_prefix": "TITLE_NAME"})
    people_df = people_df.rename(columns={"name.0.first_name": "FIRSTMID_NAME"})
    people_df = people_df.rename(columns={"name.0.last_name": "LASTSUFF_NAME"})
    people_df = people_df.rename(columns={"name.0.value": "PREFERRED_NAME"})
    people_df = people_df.rename(
        columns={"lifecycle.birth.0.date.0.value": "BIRTH_DATE"}
    )
    people_df = people_df.rename(
        columns={"lifecycle.death.0.date.0.value": "DEATH_DATE"}
    )
    people_df = people_df.rename(
        columns={"lifecycle.birth.0.place.0.summary_title": "BIRTH_PLACE"}
    )
    people_df = people_df.rename(
        columns={"lifecycle.death.0.place.0.summary_title": "DEATH_PLACE"}
    )
    people_df = people_df.rename(columns={"nationality.0": "NATIONALITY"})
    people_df = people_df.rename(columns={"description.0.value": "DESCRIPTION"})
    people_df = people_df.rename(columns={"gender": "GENDER"})

    people_df["URI"] = adlib_people_prefix + people_df["ID"].astype(str)
    people_df["BIRTH_DATE"] = people_df["BIRTH_DATE"].apply(get_year_from_date_value)
    people_df["DEATH_DATE"] = people_df["DEATH_DATE"].apply(get_year_from_date_value)
    people_df["NATIONALITY"] = people_df["NATIONALITY"].apply(
        datastore_helpers.split_list_string
    )
    people_df["NATIONALITY"] = people_df["NATIONALITY"].apply(
        lambda x: flatten_list_of_lists(
            [datastore_helpers.get_country_from_nationality(i) for i in x]
        )
    )

    people_df["BIRTH_PLACE"] = people_df["BIRTH_PLACE"].apply(
        lambda i: get_wikidata_uri_from_placename(i, False, adlib_placename_qid_mapping)
    )
    people_df["DEATH_PLACE"] = people_df["DEATH_PLACE"].apply(
        lambda i: get_wikidata_uri_from_placename(i, False, adlib_placename_qid_mapping)
    )

    # remove newlines and tab chars
    people_df.loc[:, "DESCRIPTION"] = people_df.loc[:, "DESCRIPTION"].apply(
        datastore_helpers.process_text
    )

    people_df.loc[:, "GENDER"] = people_df.loc[:, "GENDER"].replace(
        {"female": WD.Q6581072, "male": WD.Q6581097}
    )

    logger.info("loading adlib people data")
    record_loader.add_records(table_name, people_df, add_type=WD.Q5)


def create_people_disambiguating_description(row: pd.Series) -> str:
    """
    Original description col = BIOGRAPHY.
    Components:
    - NATIONALITY + OCCUPATION -> 'American photographer.'
    - BIRTH_DATE + BIRTH_PLACE -> 'Born 1962, United Kingdom.'
    - DEATH_DATE + DEATH_PLACE + CAUSE_OF_DEATH -> 'Died 1996 of heart attack.' (Add place if no overlap between
        BIRTH_PLACE and DEATH_PLACE strings. Joined to founded string above)
    - BIOGRAPHY (original description)
    NOTE: must be used before dates are converted to numbers using `get_year_from_date_string`, so that
    uncertainty such as 'about 1971' is added to the description.
    """

    # NATIONALITY + OCCUPATION (only uses first of each)
    nationality = str(row["NATIONALITY"][0])
    occupation = str(row["OCCUPATION"][0])
    add_nationality = (nationality != "nan") and (
        nationality.lower() not in row.BIOGRAPHY.lower()
    )
    add_occupation = (occupation != "nan") and (
        occupation.lower() not in row.BIOGRAPHY.lower()
    )

    if add_nationality and add_occupation:
        nationality_occupation_str = (
            f"{nationality.strip().title()} {occupation.strip()}."
        )
    elif add_nationality:
        nationality_occupation_str = f"{nationality.strip().title()}."
    elif add_occupation:
        nationality_occupation_str = f"{occupation.strip().capitalize()}."
    else:
        nationality_occupation_str = ""

    # BIRTH_PLACE + BIRTH_DATE
    add_birth_place = (str(row["BIRTH_PLACE"]) != "nan") and (
        str(row["BIRTH_PLACE"]).lower() not in row.BIOGRAPHY.lower()
    )
    add_birth_date = (str(row["BIRTH_DATE"]) != "nan") and (
        str(row["BIRTH_DATE"]).lower() not in row.BIOGRAPHY.lower()
    )

    # Also check for dates minus suffixes, e.g. 200-250 should match with 200-250 AD and vice-versa
    if re.findall(r"\d+-?\d*", str(row["BIRTH_DATE"])):
        add_birth_date = add_birth_date and (
            re.findall(r"\d+-?\d*", row["BIRTH_DATE"])[0].lower()
            not in row.BIOGRAPHY.lower()
        )

    if add_birth_place and add_birth_date:
        founded_str = f"Born in {row.BIRTH_PLACE.strip()}, {row.BIRTH_DATE.strip()}."
    elif add_birth_place:
        founded_str = f"Born in {row.BIRTH_PLACE.strip()}."
    elif add_birth_date:
        founded_str = f"Born {row.BIRTH_DATE.strip()}."
    else:
        founded_str = ""

    # DEATH_PLACE + DEATH_DATE
    add_death_place = (
        row["DEATH_PLACE"]
        and (str(row["DEATH_PLACE"]) != "nan")
        and (str(row["DEATH_PLACE"]).lower() not in row.BIOGRAPHY.lower())
        and (str(row["DEATH_PLACE"]) not in str(row["BIRTH_PLACE"]))
        and (str(row["BIRTH_PLACE"]) not in str(row["DEATH_PLACE"]))
    )
    add_death_date = (str(row["DEATH_DATE"]) != "nan") and (
        str(row["DEATH_DATE"]).lower() not in row.BIOGRAPHY.lower()
    )
    # Also check for dates minus suffixes, e.g. 200-250 should match with 200-250 AD and vice-versa
    if re.findall(r"\d+-?\d*", str(row["DEATH_DATE"])):
        add_death_date = add_death_date and (
            re.findall(r"\d+-?\d*", row["DEATH_DATE"])[0].lower()
            not in row.BIOGRAPHY.lower()
        )

    cause_of_death = str(row["CAUSE_OF_DEATH"]).strip()
    add_cause_of_death = (cause_of_death != "nan") and (
        cause_of_death.lower() not in row.BIOGRAPHY.lower()
    )
    if cause_of_death.startswith("illness (") and cause_of_death.endswith(")"):
        cause_of_death = cause_of_death.split("(")[1][0:-1]

    if add_death_place and add_death_date:
        dissolved_str = f"Died in {row.DEATH_PLACE.strip()}, {row.DEATH_DATE.strip()}."
    elif add_death_place:
        dissolved_str = f"Died in {row.DEATH_PLACE.strip()}."
    elif add_death_date:
        dissolved_str = f"Died {row.DEATH_DATE.strip()}."
    else:
        dissolved_str = ""

    if add_cause_of_death and (add_death_date or add_death_place):
        dissolved_str = (
            dissolved_str[0:-1] + " of " + row.CAUSE_OF_DEATH.lower().strip() + "."
        )
    elif add_cause_of_death:
        dissolved_str += f"Cause of death was {row.CAUSE_OF_DEATH.lower().strip()}."

    # Assemble
    dates_str = " ".join([founded_str, dissolved_str]).strip()

    # add space and full stop (if needed) to end of description
    if row.BIOGRAPHY:
        description = (
            row.BIOGRAPHY.strip()
            if row.BIOGRAPHY.strip()[-1] == "."
            else f"{row.BIOGRAPHY.strip()}."
        )
    else:
        description = ""

    # we shuffle the components of the description so any model using them does not learn the order that we put them in
    aug_description_components = [nationality_occupation_str, description, dates_str]
    random.shuffle(aug_description_components)

    return (" ".join(aug_description_components)).strip()


def load_people_data(people_data_path):
    """Load data from CSV files """

    def reverse_preferred_name(name: str) -> str:
        if not pd.isnull(name) and len(name.split(",")) == 2:
            return f"{name.split(',')[1].strip()} {name.split(',')[0].strip()}"
        else:
            return name

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
    people_df["PREFERRED_NAME"] = people_df["PREFERRED_NAME"].apply(
        reverse_preferred_name
    )
    people_df["OCCUPATION"] = people_df["OCCUPATION"].apply(
        datastore_helpers.split_list_string
    )
    people_df["NATIONALITY"] = people_df["NATIONALITY"].apply(
        datastore_helpers.split_list_string
    )
    people_df[["DESCRIPTION", "NOTE"]] = people_df[["DESCRIPTION", "NOTE"]].fillna("")

    # remove newlines and tab chars
    people_df.loc[:, ["DESCRIPTION", "NOTE"]] = people_df.loc[
        :, ["DESCRIPTION", "NOTE"]
    ].applymap(datastore_helpers.process_text)

    # create combined text fields
    newline = " \n "  # can't insert into fstring below
    people_df.loc[:, "BIOGRAPHY"] = people_df[["DESCRIPTION", "NOTE"]].apply(
        lambda x: f"{newline.join(x)}" if any(x) else "", axis=1
    )

    people_df["DISAMBIGUATING_DESCRIPTION"] = people_df.apply(
        create_people_disambiguating_description, axis=1
    )

    # all of these must happen after creating DISAMBIGUATING_DESCRIPTION as they modify the text values of fields
    # that are used
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
    people_df["BIRTH_DATE"] = people_df["BIRTH_DATE"].apply(get_year_from_date_value)
    people_df["DEATH_DATE"] = people_df["DEATH_DATE"].apply(get_year_from_date_value)
    people_df.loc[:, "GENDER"] = people_df.loc[:, "GENDER"].replace(
        {"F": WD.Q6581072, "M": WD.Q6581097}
    )

    logger.info("loading people data")
    record_loader.add_records(table_name, people_df, add_type=WD.Q5)


def load_adlib_orgs_data(adlib_people_data_path):
    # identifier in field_mapping
    table_name = "ORGANISATION_ADLIB"

    org_df = pd.read_csv(adlib_people_data_path, low_memory=False, nrows=max_records)

    # PREPROCESS
    org_df = org_df[org_df["type.type"] == "institution"]
    org_df = org_df.rename(columns={"admin.uid": "ID"})
    org_df = org_df.rename(columns={"name.0.value": "PREFERRED_NAME"})
    org_df = org_df.rename(columns={"use.0.summary_title": "SUMMARY_TITLE"})
    org_df = org_df.rename(columns={"lifecycle.birth.0.date.0.value": "BIRTH_DATE"})
    org_df = org_df.rename(columns={"lifecycle.death.0.date.0.value": "DEATH_DATE"})
    org_df = org_df.rename(columns={"nationality.0": "NATIONALITY"})
    org_df = org_df.rename(columns={"description.0.value": "DESCRIPTION"})

    org_df["PREFIX"] = people_prefix

    org_df["URI"] = org_df["ID"].apply(lambda i: adlib_people_prefix + str(i))
    # if SUMMARY_TITLE exists, use it as a label over PREFERRED_NAME, then apply PREFERRED_NAME as an alias
    org_df["LABEL"] = org_df.apply(
        lambda row: row["SUMMARY_TITLE"]
        if not pd.isnull(row["SUMMARY_TITLE"])
        else row["PREFERRED_NAME"],
        axis=1,
    )
    org_df["ALIAS"] = org_df.apply(
        lambda row: row["PREFERRED_NAME"]
        if not pd.isnull(row["SUMMARY_TITLE"])
        else "",
        axis=1,
    )

    org_df["BIRTH_DATE"] = org_df["BIRTH_DATE"].apply(get_year_from_date_value)
    org_df["DEATH_DATE"] = org_df["DEATH_DATE"].apply(get_year_from_date_value)
    org_df["NATIONALITY"] = org_df["NATIONALITY"].apply(
        datastore_helpers.split_list_string
    )
    org_df["NATIONALITY"] = org_df["NATIONALITY"].apply(
        lambda x: flatten_list_of_lists(
            [datastore_helpers.get_country_from_nationality(i) for i in x]
        )
    )

    # remove newlines and tab chars
    org_df.loc[:, "DESCRIPTION"] = org_df.loc[:, "DESCRIPTION"].apply(
        datastore_helpers.process_text
    )
    logger.info("loading adlib orgs data")
    record_loader.add_records(table_name, org_df, add_type=WD.Q43229)


def create_org_disambiguating_description(row: pd.Series) -> str:
    """
    Original description col = BIOGRAPHY.
    Components:
    - NATIONALITY + OCCUPATION -> 'British Railway Board'
    - BIRTH_DATE + BIRTH_PLACE -> 'Founded 1962, United Kingdom'
    - DEATH_DATE + DEATH_PLACE -> 'Dissolved 1996.' (Add place if no overlap between
        BIRTH_PLACE and DEATH_PLACE strings. Joined to founded string above)
    - BIOGRAPHY (original description)
    NOTE: must be used before dates are converted to numbers using `get_year_from_date_string`, so that
    uncertainty such as 'about 1971' is added to the description.
    """

    # NATIONALITY + OCCUPATION (only uses first of each)
    nationality = str(row["NATIONALITY"][0])
    occupation = str(row["OCCUPATION"][0])
    add_nationality = (nationality != "nan") and (
        nationality.lower() not in row.BIOGRAPHY.lower()
    )
    add_occupation = (occupation != "nan") and (
        occupation.lower() not in row.BIOGRAPHY.lower()
    )

    if add_nationality and add_occupation:
        nationality_occupation_str = (
            f"{nationality.strip().title()} {occupation.strip()}."
        )
    elif add_nationality:
        nationality_occupation_str = f"{nationality.strip().title()}."
    elif add_occupation:
        nationality_occupation_str = f"{occupation.strip().capitalize()}."
    else:
        nationality_occupation_str = ""

    # BIRTH_PLACE + BIRTH_DATE
    add_birth_place = (str(row["BIRTH_PLACE"]) != "nan") and (
        str(row["BIRTH_PLACE"]).lower() not in row.BIOGRAPHY.lower()
    )
    add_birth_date = (str(row["BIRTH_DATE"]) != "nan") and (
        str(row["BIRTH_DATE"]).lower() not in row.BIOGRAPHY.lower()
    )
    # Also check for dates minus suffixes, e.g. 200-250 should match with 200-250 AD and vice-versa
    if re.findall(r"\d+-?\d*", str(row["BIRTH_DATE"])):
        add_birth_date = add_birth_date and (
            re.findall(r"\d+-?\d*", row["BIRTH_DATE"])[0].lower()
            not in row.BIOGRAPHY.lower()
        )

    if add_birth_place and add_birth_date:
        founded_str = f"Founded in {row.BIRTH_PLACE.strip()}, {row.BIRTH_DATE.strip()}."
    elif add_birth_place:
        founded_str = f"Founded in {row.BIRTH_PLACE.strip()}."
    elif add_birth_date:
        founded_str = f"Founded {row.BIRTH_DATE.strip()}."
    else:
        founded_str = ""

    # DEATH_PLACE + DEATH_DATE
    add_death_place = (
        (str(row["DEATH_PLACE"]) != "nan")
        and (str(row["DEATH_PLACE"]).lower() not in row.BIOGRAPHY.lower())
        and (str(row["DEATH_PLACE"]) not in str(row["BIRTH_PLACE"]))
        and (str(row["BIRTH_PLACE"]) not in str(row["DEATH_PLACE"]))
    )
    add_death_date = (str(row["DEATH_DATE"]) != "nan") and (
        str(row["DEATH_DATE"]).lower() not in row.BIOGRAPHY.lower()
    )
    # Also check for dates minus suffixes, e.g. 200-250 should match with 200-250 AD and vice-versa
    if re.findall(r"\d+-?\d*", str(row["DEATH_DATE"])):
        add_death_date = add_death_date and (
            re.findall(r"\d+-?\d*", row["DEATH_DATE"])[0].lower()
            not in row.BIOGRAPHY.lower()
        )

    if add_death_place and add_death_date:
        dissolved_str = (
            f"Dissolved in {row.DEATH_PLACE.strip()}, {row.DEATH_DATE.strip()}."
        )
    elif add_death_place:
        dissolved_str = f"Dissolved in {row.DEATH_PLACE.strip()}."
    elif add_death_date:
        dissolved_str = f"Dissolved {row.DEATH_DATE.strip()}."
    else:
        dissolved_str = ""

    # Assemble
    dates_str = " ".join([founded_str, dissolved_str]).strip()

    # add space and full stop (if needed) to end of description
    if row.BIOGRAPHY:
        description = (
            row.BIOGRAPHY.strip()
            if row.BIOGRAPHY.strip()[-1] == "."
            else f"{row.BIOGRAPHY.strip()}."
        )
    else:
        description = ""

    # we shuffle the components of the description so any model using them does not learn the order that we put them in
    aug_description_components = [nationality_occupation_str, description, dates_str]
    random.shuffle(aug_description_components)

    return (" ".join(aug_description_components)).strip()


def load_orgs_data(people_data_path):
    # identifier in field_mapping
    table_name = "ORGANISATION"

    org_df = pd.read_csv(people_data_path, low_memory=False, nrows=max_records)
    # TODO: use isIndividual flag here
    org_df = org_df[org_df["GENDER"] == "N"]

    # PREPROCESS
    org_df["URI"] = people_prefix + org_df["LINK_ID"].astype(str)

    org_df[["DESCRIPTION", "NOTE"]] = org_df[["DESCRIPTION", "NOTE"]].fillna("")
    org_df[["DESCRIPTION", "NOTE"]] = org_df[["DESCRIPTION", "NOTE"]].applymap(
        datastore_helpers.process_text
    )
    org_df[["OCCUPATION", "NATIONALITY"]] = org_df[
        ["OCCUPATION", "NATIONALITY"]
    ].applymap(datastore_helpers.split_list_string)

    newline = " \n "  # can't insert into fstring below
    org_df.loc[:, "BIOGRAPHY"] = org_df[["DESCRIPTION", "NOTE"]].apply(
        lambda x: f"{newline.join(x)}" if any(x) else "", axis=1
    )

    org_df["DISAMBIGUATING_DESCRIPTION"] = org_df.apply(
        create_org_disambiguating_description, axis=1
    )

    org_df["NATIONALITY"] = org_df["NATIONALITY"].apply(
        lambda x: flatten_list_of_lists(
            [datastore_helpers.get_country_from_nationality(i) for i in x]
        )
    )

    org_df["BIRTH_DATE"] = org_df["BIRTH_DATE"].apply(get_year_from_date_value)
    org_df["DEATH_DATE"] = org_df["DEATH_DATE"].apply(get_year_from_date_value)

    logger.info("loading orgs data")
    record_loader.add_records(table_name, org_df)

    # also add type organization (Q43229)
    org_df["type_org"] = qid_to_url("Q43229")
    record_loader.add_triples(
        org_df, RDF.type, subject_col="URI", object_col="type_org"
    )

    return


def load_adlib_mimsy_join_people_orgs(join_table_path):
    join_df = pd.read_csv(join_table_path)
    join_df["mimsy_url"] = people_prefix + join_df["mimsy_id"].astype(str)
    join_df["adlib_url"] = adlib_people_prefix + join_df["adlib_id"].astype(str)

    record_loader.add_triples(
        join_df, FOAF.page, subject_col="mimsy_url", object_col="adlib_url"
    )
    record_loader.add_triples(
        join_df, FOAF.page, subject_col="adlib_url", object_col="mimsy_url"
    )


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
    # use FOAF.maker (with inverse FOAF.made) for people
    people_makers = maker_df[maker_df["GENDER"].isin(["M", "F"])]
    record_loader.add_triples(
        people_makers, FOAF.maker, subject_col="OBJECT_ID", object_col="PERSON_ORG_ID"
    )
    record_loader.add_triples(
        people_makers, FOAF.made, subject_col="PERSON_ORG_ID", object_col="OBJECT_ID"
    )

    # where we don't have gender information use FOAF.maker as it's human-readable
    undefined_makers = maker_df[maker_df["GENDER"].isna()]
    record_loader.add_triples(
        undefined_makers,
        FOAF.maker,
        subject_col="OBJECT_ID",
        object_col="PERSON_ORG_ID",
    )
    record_loader.add_triples(
        undefined_makers, FOAF.made, subject_col="PERSON_ORG_ID", object_col="OBJECT_ID"
    )

    # use 'product or material produced' Wikidata property for organisations, and FOAF.maker for the inverse
    orgs_makers = maker_df[maker_df["GENDER"] == "N"]
    record_loader.add_triples(
        orgs_makers, WDT.P1056, subject_col="PERSON_ORG_ID", object_col="OBJECT_ID"
    )
    record_loader.add_triples(
        orgs_makers, FOAF.maker, subject_col="OBJECT_ID", object_col="PERSON_ORG_ID"
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
        user_df, SKOS.related, subject_col="SUBJECT", object_col="OBJECT"
    )
    # used by
    record_loader.add_triples(
        user_df, SKOS.related, subject_col="OBJECT", object_col="SUBJECT"
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


def preprocess_text_for_ner(text: str) -> str:
    # remove URLs
    text = re.sub(
        r"((?:https?://|www\.|https?://|www\.)[a-z0-9\.:].*?(?=[\s;,!:\[\]]|$))",
        "",
        text,
    )

    # remove dois, e.g. doi:10.1093/ref:odnb/9153
    text = re.sub(r"doi:[\w.:/\\;]*\b", "", text)

    # remove any text in normal brackets
    #     text = re.sub(r"\([^()]*\)", "", text)

    #     # remove any text in square brackets
    #     text = re.sub(r"\[[^\[\]]*\]", "", text)

    # replace newline characters with spaces
    text = text.replace("\n", " ")

    # remove strings in `strings_to_remove`
    strings_to_remove = [
        "WIKI:",
        "WIKI",
        "REF:",
        "VIAF:",
        "Oxford Dictionary of National Biography",
        "Oxford University Press",
        "Oxford Dictionary of National Biography, Oxford University Press",
        "Library of Congress Authorities:",
    ]
    strings_to_remove.sort(key=len, reverse=True)
    for s in strings_to_remove:
        text = text.replace(s, "")

    # finally, remove any leading or trailing whitespace
    text = text.strip()

    return text


def load_ner_annotations(
    model_type: str,
    use_trained_linker: bool,
    nel_training_data_path: Optional[str] = None,
    linking_confidence_threshold: float = 0.8,
):
    """
    Args:
        model_type (str): spacy model type e.g. "en_core_web_trf"
        use_trained_linker (bool): whether to use trained entity linker to add links to graph (True), or
            export training data to train an entity linker (False)
        nel_training_data_path (Optional[str], optional): Path to training data Excel file for linker, either to train it, or where it's exported. Defaults to None.
        linking_confidence_threshold (float, optional): Threshold for linker. Defaults to 0.8.
    """

    source_description_field = (
        target_description_field
    ) = "data.http://www.w3.org/2001/XMLSchema#description"  # "data.https://schema.org/disambiguatingDescription"
    target_title_field = "graph.@rdfs:label.@value"
    target_alias_field = "graph.@skos:altLabel.@value"
    target_type_field = "graph.@skos:hasTopConcept.@value"

    ner_loader = datastore.NERLoader(
        record_loader,
        source_es_index=config.ELASTIC_SEARCH_INDEX,
        source_description_field=source_description_field,
        target_es_index=config.ELASTIC_SEARCH_INDEX,
        target_title_field=target_title_field,
        target_description_field=target_description_field,
        target_type_field=target_type_field,
        target_alias_field=target_alias_field,
        text_preprocess_func=preprocess_text_for_ner,
        entity_types_to_link={
            "PERSON",
            "OBJECT",
            "ORG",
        },
    )

    _ = ner_loader.get_list_of_entities_from_source_index(
        model_type, spacy_batch_size=16
    )
    ner_loader.get_link_candidates_from_target_index(candidates_per_entity_mention=10)

    if use_trained_linker:
        # load NEL training data
        print(f"Using NEL training data from {nel_training_data_path}")
        df = pd.read_excel(nel_training_data_path, index_col=0)
        df.loc[~df["link_correct"].isnull(), "link_correct"] = df.loc[
            ~df["link_correct"].isnull(), "link_correct"
        ].apply(int)
        nel_train_data = df[
            (~df["link_correct"].isnull()) & (df["candidate_rank"] != -1)
        ]
        ner_loader.train_entity_linker(nel_train_data)
    else:
        # get NEL training data to annotate
        links_data = ner_loader.get_links_data_for_review()
        links_data.to_excel(nel_training_data_path)
        print(f"NEL training data exported to {nel_training_data_path}")

    ner_loader.load_entities_into_source_index(
        linking_confidence_threshold, batch_size=32768
    )


if __name__ == "__main__":
    people_data_path = "../GITIGNORE_DATA/smg-datasets-private/mimsy-people-export.csv"
    object_data_path = (
        "../GITIGNORE_DATA/smg-datasets-private/mimsy-catalogue-export.csv"
    )
    adlib_data_path = "s3://smg-datasets/adlib-document-dump-relevant-columns.csv"
    adlib_people_data_path = "s3://heritageconnector/adlib-people-dump.csv"
    maker_data_path = "../GITIGNORE_DATA/smg-datasets-private/items_makers.csv"
    user_data_path = "../GITIGNORE_DATA/smg-datasets-private/items_users.csv"
    mimsy_adlib_join_data_path = "../GITIGNORE_DATA/mimsy_adlib_link_table.csv"

    # ---

    datastore.create_index()
    load_people_data(people_data_path)
    load_adlib_people_data(adlib_people_data_path)
    load_orgs_data(people_data_path)
    load_adlib_orgs_data(adlib_people_data_path)
    load_adlib_mimsy_join_people_orgs(mimsy_adlib_join_data_path)
    load_object_data(object_data_path)
    load_adlib_document_data(adlib_data_path)
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
    # # for running using a trained linker
    load_ner_annotations(
        "en_core_web_trf",
        use_trained_linker=True,
        nel_training_data_path="../GITIGNORE_DATA/NEL/review_data_1103.xlsx",
    )
    # # for running to produce unlabelled training data at `nel_training_data_path`
    # # load_ner_annotations(
    # #     "en_core_web_trf",
    # #     use_trained_linker=False,
    # #     nel_training_data_path=f"../GITIGNORE_DATA/NEL/nel_train_data_{get_timestamp()}.xlsx",
    # # )
