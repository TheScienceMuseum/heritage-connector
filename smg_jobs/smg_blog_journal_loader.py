"""Loading blog and journal data scraped using [https://github.com/TheScienceMuseum/journal-blog-scraper/]
"""

import sys

sys.path.append("..")

from heritageconnector.config import config, field_mapping
from heritageconnector import datastore, datastore_helpers
from heritageconnector.namespace import SDO, RDF
from heritageconnector.utils.generic import flatten_list_of_lists
from heritageconnector import logging
from smg_jobs.smg_loader import preprocess_text_for_ner, load_nel_training_data

import pandas as pd
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


def load_blog_data(blog_data_path):
    blog_df = pd.read_json(blog_data_path)
    blog_df["links"] = blog_df["links"].apply(
        lambda i: flatten_list_of_lists(i.values())
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
    journal_df = journal_df.rename(columns={"url": "URI"})
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
    print(f"Using NEL training data from {nel_training_data_path}")
    nel_train_data = load_nel_training_data(nel_training_data_path)
    ner_loader.train_entity_linker(nel_train_data)

    ner_loader.load_entities_into_source_index(
        linking_confidence_threshold,
        batch_size=32768,
    )


if __name__ == "__main__":
    # PARAMETERS
    JOURNAL_NO_PARAGRAPHS = 2

    # CORE DATA
    datastore.create_index(es_indices["blog"])
    load_blog_data(
        blog_data_path="/Users/kalyan/Documents/SMG/journal-blog-scraper/output_data/blog.json"
    )

    datastore.create_index(es_indices["journal"])
    load_journal_data(
        journal_data_path="/Users/kalyan/Documents/SMG/journal-blog-scraper/output_data/journal.json"
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

    # TODO: BLINK
