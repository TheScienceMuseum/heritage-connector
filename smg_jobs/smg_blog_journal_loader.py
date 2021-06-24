"""Loading blog and journal data scraped using [https://github.com/TheScienceMuseum/journal-blog-scraper/]
"""

import sys

sys.path.append("..")

from heritageconnector.config import config, field_mapping
from heritageconnector import datastore, datastore_helpers
from heritageconnector.namespace import SDO, RDF
from heritageconnector.utils.generic import flatten_list_of_lists, get_timestamp
from heritageconnector import logging
from heritageconnector.nlp.nel import BLINKServiceWrapper
from smg_jobs.smg_loader import preprocess_text_for_ner, load_nel_training_data

import pandas as pd
import re
import unicodedata

logger = logging.get_logger(__name__)

# disable pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

es_indices = {"blog": "heritageconnector_blog", "journal": "heritageconnector_journal"}

record_loaders = {
    "blog": datastore.RecordLoader("SMG", field_mapping, es_indices["blog"]),
    "journal": datastore.RecordLoader("SMG", field_mapping, es_indices["journal"]),
}


def process_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)

    return text


def normalise_collection_url(url: str) -> str:
    """Remove anything after ID (e.g. `cp43213`) and change 'sciencemuseum.org.uk' to 'sciencemuseumgroup.org.uk'"""

    url = url.replace(
        "collection.sciencemuseum.org.uk", "collection.sciencemuseumgroup.org.uk"
    )

    if "collection.sciencemuseumgroup.org.uk" in url:
        url = re.findall(r"https://(?:\w.+)/(?:co|cp|ap|aa)(?:\d+)", url)[0]

    return url


def load_blog_data(blog_data_path):
    blog_df = pd.read_json(blog_data_path)
    # blog_df = blog_df.head(100)  # for debugging
    blog_df["links"] = (
        blog_df["links"]
        .apply(lambda i: flatten_list_of_lists(i.values()))
        .apply(lambda url_list: [normalise_collection_url(url) for url in url_list])
    )
    blog_df = blog_df.rename(columns={"url": "URI"})
    blog_df["text_by_paragraph"] = blog_df["text_by_paragraph"].apply("\n".join)
    blog_df[["caption", "text_by_paragraph"]] = blog_df[
        ["caption", "text_by_paragraph"]
    ].applymap(lambda i: process_text(i) if i else i)
    blog_df[["categories", "tags"]] = blog_df[["categories", "tags"]].applymap(
        lambda lst: [i.lower() for i in lst]
    )

    logger.info("loading blog data")
    record_loaders["blog"].add_records("BLOG_POST", blog_df)


def load_journal_data(journal_data_path):
    journal_df = pd.read_json(journal_data_path)
    # journal_df = journal_df.head(100)  # for debugging
    journal_df = journal_df.rename(columns={"url": "URI"})
    journal_df["abstract"] = journal_df["abstract"].fillna("")
    journal_df["text_by_paragraph"] = journal_df.apply(
        lambda row: [row["abstract"]] + row["text_by_paragraph"], axis=1
    )
    journal_df["text_by_paragraph"] = (
        journal_df["text_by_paragraph"]
        .apply(lambda i: "\n".join(i[:JOURNAL_NO_PARAGRAPHS]))
        .apply(process_text)
    )
    journal_df[["keywords", "tags"]] = journal_df[["keywords", "tags"]].applymap(
        lambda lst: [i.lower() for i in lst]
    )

    logger.info("loading journal data")
    record_loaders["journal"].add_records("JOURNAL_ARTICLE", journal_df)


def load_ner_annotations(
    blog_or_journal: str,
    model_type: str,
    nel_training_data_path: str = None,
    linking_confidence_threshold: float = 0.75,
):
    logger.info(f"Loading NER annotations for {blog_or_journal}")
    ner_loader_kwargs = dict(
        target_es_index="heritageconnector",
        target_context_field="data.https://schema.org/disambiguatingDescription",
        target_title_field="graph.@rdfs:label.@value",
        target_alias_field="graph.@skos:altLabel.@value",
        target_type_field="graph.@skos:hasTopConcept.@value",
        target_record_types=("PERSON", "OBJECT", "ORGANISATION"),
        text_preprocess_func=preprocess_text_for_ner,
    )

    ner_loader = datastore.NERLoader(
        record_loaders[blog_or_journal],
        source_es_index=es_indices[blog_or_journal],
        source_description_field="data.https://schema.org/text",
        **ner_loader_kwargs,
    )

    ner_loader.get_list_of_entities_from_source_index(model_type, spacy_batch_size=16)

    ner_loader.get_link_candidates_from_target_index(candidates_per_entity_mention=15)

    # load NEL training data
    logger.info(f"Using NEL training data from {nel_training_data_path}")
    nel_train_data = load_nel_training_data(nel_training_data_path)
    ner_loader.train_entity_linker(nel_train_data)

    ner_loader.load_entities_into_source_index(
        linking_confidence_threshold,
        batch_size=32768,
    )


def create_blink_json(
    blog_or_journal: str,
    output_path: str,
    description_field: str = "data.https://schema.org/text",
    blink_threshold=0.8,
    blink_base_url="localhost",
):
    """Run BLINK on entities in the blog or journal and export the results to a JSON file."""
    logger.info(f"Running BLINK for {blog_or_journal}")

    endpoint = "http://" + blink_base_url + ":8000/blink/multiple"

    entity_fields = [
        "graph.@hc:entityPERSON.@value",
        "graph.@hc:entityORG.@value",
        "graph.@hc:entityLOC.@value",
        "graph.@hc:entityFAC.@value",
        "graph.@hc:entityOBJECT.@value",
        "graph.@hc:entityLANGUAGE.@value",
        "graph.@hc:entityNORP.@value",
        "graph.@hc:entityEVENT.@value",
        # "graph.@hc:entityDATE.@value",
    ]

    # NOTE: the final importing threshold is 0.9 by default
    blink_threshold = 0.8

    blink_service = BLINKServiceWrapper(
        endpoint,
        description_field=description_field,
        entity_fields=entity_fields,
        wiki_link_threshold=blink_threshold,
    )

    blink_service.process_unlinked_entity_mentions(
        es_indices[blog_or_journal],
        output_path,
        page_size=12,
        limit=None,
    )


if __name__ == "__main__":
    # PARAMETERS
    JOURNAL_NO_PARAGRAPHS = 3

    # CORE DATA
    datastore.create_index(es_indices["blog"])
    load_blog_data(blog_data_path="../../journal-blog-scraper/output_data/blog.json")

    datastore.create_index(es_indices["journal"])
    load_journal_data(
        journal_data_path="../../journal-blog-scraper/output_data/journal.json"
    )

    # NER & NEL
    model_type = "en_core_web_trf"
    nel_training_data_path = "../GITIGNORE_DATA/NEL/nel_train_data_20210610-1035_combined_with_review_data_fixed.xlsx"
    load_ner_annotations(
        "blog", model_type, nel_training_data_path=nel_training_data_path
    )
    load_ner_annotations(
        "journal", model_type, nel_training_data_path=nel_training_data_path
    )

    # BLINK - predict
    blog_blink_output_path = (
        f"../GITIGNORE_DATA/blink_output_blog_{get_timestamp()}.jsonl"
    )
    create_blink_json("blog", blog_blink_output_path)

    journal_blink_output_path = (
        f"../GITIGNORE_DATA/blink_output_journal_{get_timestamp()}.jsonl"
    )
    create_blink_json("journal", journal_blink_output_path)

    # BLINK - load
    blink_loader = datastore.BLINKLoader()
    blink_loader.load_blink_results_to_es_from_json(
        blog_blink_output_path, es_indices["blog"]
    )
    blink_loader.load_blink_results_to_es_from_json(
        journal_blink_output_path, es_indices["journal"]
    )
