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

import pandas as pd
import random
import re
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
from heritageconnector.utils.generic import get_timestamp
from heritageconnector import logging

logger = logging.get_logger(__name__)

# disable pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# optional limit of number of records to import to test loader. no limit -> None
# passed as an argument into `pd.read_csv`. You might want to use your own implementation
# depending on your source data format
max_records = None
MAX_NO_WORDS_PER_DESCRIPTION = 500

# create instance of RecordLoader from datastore
record_loader = datastore.RecordLoader(
    collection_name="vanda", field_mapping=field_mapping
)

#  ======================================================

## Content Table Loading


def trim_description(desc: str, n_words: int) -> str:
    """Return the first `n_words` words of description `desc`/"""

    return " ".join(str(desc).split(" ")[0:n_words])


def reverse_person_preferred_name_and_strip_brackets(name: str) -> str:
    name_stripped = re.sub(r"\([^()]*\)", "", name)

    if not pd.isnull(name_stripped) and len(name_stripped.split(",")) == 2:
        return f"{name_stripped.split(',')[1].strip()} {name_stripped.split(',')[0].strip()}"
    else:
        return name_stripped


def create_object_disambiguating_description(
    row: pd.Series, description_field: str = "COMBINED_DESCRIPTION"
) -> str:
    """
    Original description col = `description_field` (function argument).
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
        and (str(row.OBJECT_TYPE).lower() not in row[description_field].lower())
        and (
            str(row.OBJECT_TYPE).rstrip("s").lower()
            not in row[description_field].lower()
        )
        and (
            str(row.OBJECT_TYPE).rstrip("es").lower()
            not in row[description_field].lower()
        )
    ):
        object_type = f"{row.OBJECT_TYPE.capitalize().strip()}."
    else:
        object_type = ""

    # PRIMARY_PLACE + PRIMARY_DATE
    add_place_made = (
        (row["PRIMARY_PLACE"])
        and (str(row["PRIMARY_PLACE"]) != "nan")
        and (str(row["PRIMARY_PLACE"]).lower() not in row[description_field].lower())
    )

    add_date_made = (
        (row["PRIMARY_DATE"])
        and (str(row["PRIMARY_DATE"]) != "nan")
        and (str(row["PRIMARY_DATE"]))
        and (str(row["PRIMARY_DATE"]) not in row[description_field].lower())
    )
    # Also check for dates minus suffixes, e.g. 200-250 should match with 200-250 AD and vice-versa
    if re.findall(r"\d+-?\d*", str(row["PRIMARY_DATE"])):
        add_date_made = add_date_made and (
            re.findall(r"\d+-?\d*", row["PRIMARY_DATE"])[0].lower()
            not in row[description_field].lower()
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
    if row[description_field].strip():
        description = (
            row[description_field].strip()
            if row[description_field].strip()[-1] == "."
            else f"{row[description_field].strip()}."
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
    object_df[["DESCRIPTION", "PHYS_DESCRIPTION", "PRODUCTION_TYPE"]] = object_df[
        ["DESCRIPTION", "PHYS_DESCRIPTION", "PRODUCTION_TYPE"]
    ].fillna("")

    # remove newlines and tab chars
    object_df.loc[
        :, ["DESCRIPTION", "PHYS_DESCRIPTION", "PRODUCTION_TYPE"]
    ] = object_df.loc[
        :, ["DESCRIPTION", "PHYS_DESCRIPTION", "PRODUCTION_TYPE"]
    ].applymap(
        datastore_helpers.process_text
    )

    # create combined text fields
    newline = " \n "  # can't insert into fstring below
    object_df.loc[:, "COMBINED_DESCRIPTION"] = object_df[
        ["DESCRIPTION", "PHYS_DESCRIPTION", "PRODUCTION_TYPE"]
    ].apply(lambda x: f"{newline.join(x)}" if any(x) else "", axis=1)
    object_df["COMBINED_DESCRIPTION"] = object_df["COMBINED_DESCRIPTION"].apply(
        lambda x: trim_description(x, MAX_NO_WORDS_PER_DESCRIPTION)
    )

    object_df["DISAMBIGUATING_DESCRIPTION"] = object_df.apply(
        create_object_disambiguating_description, axis=1
    )

    #  convert date created to year
    object_df["PRIMARY_DATE"] = object_df["PRIMARY_DATE"].apply(
        get_year_from_date_value
    )

    logger.info("loading object data")
    record_loader.add_records(table_name, object_df, add_type=WD.Q488383)

    return


def load_person_data(data_path):
    """Load data from ndjson files """

    table_name = "PERSON"
    person_df = pd.read_json(data_path, lines=True, nrows=max_records)
    person_df["BIOGRAPHY"] = person_df["BIOGRAPHY"].apply(
        lambda x: trim_description(x, MAX_NO_WORDS_PER_DESCRIPTION)
    )
    person_df["DISAMBIGUATING_DESCRIPTION"] = person_df["BIOGRAPHY"].copy()

    #  convert birthdate to year
    person_df["BIRTHDATE_EARLIEST"] = person_df["BIRTHDATE_EARLIEST"].apply(
        lambda x: int(x[0:4]) if x is not None else x
    )

    logger.info("loading person data")
    record_loader.add_records(table_name, person_df, add_type=WD.Q5)

    return


def load_org_data(data_path):
    """Load data from ndjson files """

    table_name = "ORGANISATION"
    org_df = pd.read_json(data_path, lines=True, nrows=max_records)
    org_df["HISTORY"] = org_df["HISTORY"].apply(
        lambda x: trim_description(x, MAX_NO_WORDS_PER_DESCRIPTION)
    )
    org_df["DISAMBIGUATING_DESCRIPTION"] = org_df["HISTORY"].copy()

    #  convert founding date to year
    org_df["FOUNDATION_DATE_EARLIEST"] = org_df["FOUNDATION_DATE_EARLIEST"].apply(
        lambda x: int(x[0:4]) if x is not None else x
    )

    logger.info("loading org data")
    record_loader.add_records(table_name, org_df, add_type=WD.Q43229)

    return


def load_event_data(data_path):
    """Load data from ndjson files """

    table_name = "EVENT"
    event_df = pd.read_json(data_path, lines=True, nrows=max_records)

    #  convert date created to year
    event_df[["DATE_EARLIEST", "DATE_LATEST"]] = event_df[
        ["DATE_EARLIEST", "DATE_LATEST"]
    ].applymap(get_year_from_date_value)

    logger.info("loading event data")
    record_loader.add_records(table_name, event_df, add_type=WD.P793)

    return


## Join Table Loading


def load_join_data(data_path):
    """Load subject-object-predicate triple from ndjson files and add to existing records"""
    join_df = pd.read_json(data_path, lines=True, nrows=max_records)
    join_df = join_df.rename(columns={"URI_1": "SUBJECT", "URI_2": "OBJECT"})

    logger.info("loading maker data (made by & made)")
    maker_df = join_df[join_df["relationship"] == "made_by"]
    manufactured_df = join_df[join_df["relationship"] == "manufactured_by"]
    made_df = pd.concat([maker_df, manufactured_df])
    # made
    record_loader.add_triples(
        made_df, predicate=FOAF.maker, subject_col="SUBJECT", object_col="OBJECT"
    )
    # made by
    record_loader.add_triples(
        made_df, predicate=FOAF.made, subject_col="OBJECT", object_col="SUBJECT"
    )

    logger.info("loading depicts data (depicts and depicted)")
    depicts_df = join_df[join_df["relationship"] == "depicts"]
    # depicts - A thing depicted in this representation
    record_loader.add_triples(
        depicts_df, predicate=FOAF.depicts, subject_col="SUBJECT", object_col="OBJECT"
    )
    # depiction - A depiction of some thing
    record_loader.add_triples(
        depicts_df, predicate=FOAF.depiction, subject_col="OBJECT", object_col="SUBJECT"
    )

    logger.info("loading associated data (significant_person and significant_to)")
    associate_df = join_df[join_df["relationship"] == "associated_with"]
    # significant_person - person linked to the item in any possible way
    record_loader.add_triples(
        associate_df, predicate=WDT.P3342, subject_col="SUBJECT", object_col="OBJECT"
    )

    logger.info("loading materials data (made from material and uses_this_material)")
    materials_df = join_df[join_df["relationship"] == "made_from_material"]
    # made_from_material
    record_loader.add_triples(
        materials_df, predicate=SDO.material, subject_col="SUBJECT", object_col="OBJECT"
    )

    logger.info("loading techniques data (fabrication_method)")
    technique_df = join_df[join_df["relationship"] == "fabrication_method"]
    # fabrication_method
    # NOTE: these triples don't load in as fabrication methods don't have associated documents.
    # These are the only triples we have this issue for, as all the rest are loaded between entities whose
    # types are documents (PERSON, ORGANISATION, OBJECT, EVENT). To load these triples in, we'd need to
    # make documents for each fabrication method.
    # record_loader.add_triples(
    #     # predicate = product or material produced (method -> object)
    #     technique_df, predicate=WDT.P1056, subject_col="OBJECT", object_col="SUBJECT"
    # )

    record_loader.add_triples(
        # predicate = fabrication method (object -> method)
        technique_df,
        predicate=WDT.P2079,
        subject_col="SUBJECT",
        object_col="OBJECT",
    )

    logger.info("loading events data (significant_event)")
    events_df = join_df[join_df["relationship"] == "significant_event"]
    # significant event
    record_loader.add_triples(
        # predicate = significant event (object -> event)
        events_df,
        predicate=WDT.P793,
        subject_col="SUBJECT",
        object_col="OBJECT",
    )

    return


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


def load_nel_training_data(nel_training_data_path: str) -> pd.DataFrame:
    """Load NEL training data from Excel file and return dataframe."""
    df = pd.read_excel(nel_training_data_path, index_col=0)
    df.loc[~df["link_correct"].isnull(), "link_correct"] = df.loc[
        ~df["link_correct"].isnull(), "link_correct"
    ].apply(int)
    nel_train_data = df[(~df["link_correct"].isnull()) & (df["candidate_rank"] != -1)]

    return nel_train_data


def load_ner_annotations(
    model_type: str,
    use_trained_linker: bool,
    entity_list_save_path: str = None,
    entity_list_data_path: str = None,
    nel_training_data_path: str = None,
    linking_confidence_threshold: float = 0.75,
):
    """
    Args:
        model_type (str): spacy model type e.g. "en_core_web_trf"
        use_trained_linker (bool): whether to use trained entity linker to add links to graph (True), or
            export training data to train an entity linker (False)
        entity_list_save_path (str, optional): Path to save entity list to.
        entity_list_data_path (str, optional): Path to load entity list from. Means NER can be skipped.
        nel_training_data_path (str, optional): Path to training data Excel file for linker, either to train it, or where it's exported. Defaults to None.
        linking_confidence_threshold (float, optional): Threshold for linker. Defaults to 0.8.
    """

    source_description_field = "data.http://www.w3.org/2001/XMLSchema#description"
    target_context_field = (
        source_context_field
    ) = "data.https://schema.org/disambiguatingDescription"
    target_title_field = "graph.@rdfs:label.@value"
    target_alias_field = "graph.@skos:altLabel.@value"
    target_type_field = "graph.@skos:hasTopConcept.@value"

    ner_loader = datastore.NERLoader(
        record_loader,
        source_es_index=config.ELASTIC_SEARCH_INDEX,
        source_description_field=source_description_field,
        source_context_field=source_context_field,
        target_es_index=config.ELASTIC_SEARCH_INDEX,
        target_title_field=target_title_field,
        target_context_field=target_context_field,
        target_type_field=target_type_field,
        target_alias_field=target_alias_field,
        entity_types_to_link={
            "PERSON",
            "OBJECT",
            "ORG",
        },
        target_record_types=("PERSON", "OBJECT", "ORGANISATION"),
        text_preprocess_func=preprocess_text_for_ner,
    )

    if entity_list_data_path:
        # we assume that the entity list JSON does not contain link candidates,
        # i.e. that `include_link_candidates` was False in `export_entity_list_to_json`
        ner_loader.import_entity_data_from_json(entity_list_data_path)
    else:
        ner_loader.get_list_of_entities_from_source_index(
            model_type, spacy_batch_size=16
        )

    if ner_loader.has_link_candidates:
        logger.info(
            "Link candidates present in imported entity list. Skipping getting link candidates."
        )
    else:
        ner_loader.get_link_candidates_from_target_index(
            candidates_per_entity_mention=15
        )

    if use_trained_linker:
        # load NEL training data
        logger.info(f"Using NEL training data from {nel_training_data_path}")
        nel_train_data = load_nel_training_data(nel_training_data_path)
        ner_loader.train_entity_linker(nel_train_data)
    else:
        # also optionally save list of entities
        if entity_list_save_path:
            logger.info("Exporting entity list to JSON")
            ner_loader.export_entity_data_to_json(
                # NOTE: usually (not in debug) we don't export link candidates
                entity_list_save_path,
                include_link_candidates=True,
            )

        # get NEL training data to annotate
        logger.info("Getting links data for review")
        links_data = ner_loader.get_links_data_for_review(max_no_entities=10000)
        links_data.to_excel(nel_training_data_path)
        logger.info(f"NEL training data exported to {nel_training_data_path}")

    ner_loader.load_entities_into_source_index(
        linking_confidence_threshold,
        batch_size=32768,
        force_load_without_linker=not (use_trained_linker),
    )


if __name__ == "__main__":
    object_data_path = (
        "../GITIGNORE_DATA/vanda_hc_data/hc_import/content/20210705/objects.ndjson"
    )
    person_data_path = (
        "../GITIGNORE_DATA/vanda_hc_data/hc_import/content/20210705/persons.ndjson"
    )
    org_data_path = "../GITIGNORE_DATA/vanda_hc_data/hc_import/content/20210705/organisations.ndjson"
    event_data_path = (
        "../GITIGNORE_DATA/vanda_hc_data/hc_import/content/20210705/events.ndjson"
    )
    join_data_path = (
        "../GITIGNORE_DATA/vanda_hc_data/hc_import/join/20210705/joins.ndjson"
    )

    ner_data_path = "../GITIGNORE_DATA/vanda_hc_data/NER/entity_json_2021_07_29.json"

    datastore.create_index()

    load_object_data(object_data_path)
    load_person_data(person_data_path)
    load_org_data(org_data_path)
    load_event_data(event_data_path)
    load_join_data(join_data_path)

    # for running using a trained linker
    # entity_list_data_path = "../GITIGNORE_DATA/NEL/entity_list_20210610-1035.json"
    # load_ner_annotations(
    #     "en_core_web_trf",
    #     use_trained_linker=True,
    #     entity_list_data_path=entity_list_data_path,
    #     nel_training_data_path="../GITIGNORE_DATA/NEL/nel_train_data_20210610-1035_combined_with_review_data_fixed.xlsx",
    # )
    # for running to produce unlabelled training data at `nel_training_data_path`
    load_ner_annotations(
        "en_core_web_trf",
        use_trained_linker=False,
        entity_list_save_path=f"../GITIGNORE_DATA/vanda_hc_data/NEL/entity_list_{get_timestamp()}.json",
        nel_training_data_path=f"../GITIGNORE_DATA/vanda_hc_data/NEL/nel_train_data_{get_timestamp()}.xlsx",
    )
