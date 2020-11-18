import sys

sys.path.append("..")

import os
import numpy as np
import csv
import click
from heritageconnector.disambiguation.pipelines import Disambiguator


# @click.command()
# @click.argument("train_or_test", nargs=1, type=click.Choice(["train", "test"]))
# @click.option("--output_folder", "-o", type=click.Path(exists=True))
# @click.option("--table_name", "-t", type=str)
# @click.option("--limit", "-l", type=int, default=None)
def cli(train_or_test, output_folder, table_name, limit):
    page_size = 100
    search_limit = 30

    if train_or_test == "train":
        d = Disambiguator(table_name)
        d.save_training_data_to_folder(output_folder, limit, page_size, search_limit)

    elif train_or_test == "test":
        d = Disambiguator(
            table_name,
            # extra_sparql_lines="?item sdo:isPartOf ?collection. FILTER (?collection IN ('SCM - Computing & Data Processing', 'SCM - Space Technology')).",
            # extra_sparql_lines="?item sdo:isPartOf ?collection. FILTER (?collection IN ('SCM - Art', 'NRM - Locomotives and Rolling Stock')).",
            # extra_sparql_lines="?item sdo:isPartOf ?collection. FILTER (?collection IN ('NSMM - Photographic Technology', 'SCM - Aeronautics')).",
            extra_sparql_lines="?item sdo:isPartOf ?collection. FILTER (?collection IN ('NRM - Locomotives and Rolling Stock')).",
        )
        d.save_test_data_to_folder(output_folder, limit, page_size, search_limit)


if __name__ == "__main__":
    # cli()

    # DEBUG
    # cli("train", "/Volumes/Kalyan_SSD/SMG/disambiguation/objects_131120/train_30", "OBJECT", None)
    # cli("test", "/Volumes/Kalyan_SSD/SMG/disambiguation/objects_131120/test_art_locomotives_and_rolling_stock", "OBJECT", None)
    cli(
        "test",
        "/Volumes/Kalyan_SSD/SMG/disambiguation/objects_131120/test_locomotives_and_rolling_stock",
        "OBJECT",
        None,
    )
