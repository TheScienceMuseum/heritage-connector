"""Loading blog and journal data scraped using [https://github.com/TheScienceMuseum/journal-blog-scraper/]
"""

import sys

sys.path.append("..")

from heritageconnector.config import config, field_mapping
from heritageconnector import datastore, datastore_helpers
from heritageconnector.namespace import SDO, RDF
from heritageconnector.utils.generic import flatten_list_of_lists
from heritageconnector import logging

import pandas as pd

logger = logging.get_logger(__name__)

# disable pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

blog_index = "heritageconnector_blog"
journal_index = "heritageconnector_journal"

blog_record_loader = datastore.RecordLoader("SMG", field_mapping, blog_index)
journal_record_loader = datastore.RecordLoader("SMG", field_mapping, journal_index)


def load_blog_data(blog_data_path):
    blog_df = pd.read_json(blog_data_path)
    blog_df["links"] = blog_df["links"].apply(
        lambda i: flatten_list_of_lists(i.values())
    )
    blog_df = blog_df.rename(columns={"url": "URI"})
    blog_df["text_by_paragraph"] = blog_df["text_by_paragraph"].apply("\n".join)
    blog_df[["categories", "tags"]] = blog_df[["categories", "tags"]].applymap(
        lambda lst: [i.lower() for i in lst]
    )

    logger.info("loading blog data")
    blog_record_loader.add_records("BLOG_POST", blog_df)


def load_journal_data(journal_data_path):
    journal_df = pd.read_json(journal_data_path)
    journal_df = journal_df.rename(columns={"url": "URI"})
    journal_df["text_by_paragraph"] = journal_df["text_by_paragraph"].apply("\n".join)
    journal_df[["keywords", "tags"]] = journal_df[["keywords", "tags"]].applymap(
        lambda lst: [i.lower() for i in lst]
    )

    logger.info("loading journal data")
    journal_record_loader.add_records("JOURNAL_ARTICLE", journal_df)


if __name__ == "__main__":
    datastore.create_index(blog_index)
    load_blog_data(
        blog_data_path="/Users/kalyan/Documents/SMG/journal-blog-scraper/output_data/blog.json"
    )

    datastore.create_index(journal_index)
    load_journal_data(
        journal_data_path="/Users/kalyan/Documents/SMG/journal-blog-scraper/output_data/journal.json"
    )
