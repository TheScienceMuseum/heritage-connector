import sys

sys.path.append("..")

import os
import numpy as np
import csv
import click
from heritageconnector.disambiguation.pipelines import Disambiguator


@click.command()
@click.argument("train_or_test", nargs=1, type=click.Choice(["train", "test"]))
@click.option("--output_folder", "-o", type=click.Path(exists=True))
@click.option("--table_name", "-t", type=str)
@click.option("--limit", "-l", type=int, default=None)
def cli(train_or_test, output_folder, table_name, limit):
    page_size = 100
    search_limit = 20

    d = Disambiguator()

    if train_or_test == "train":
        d.save_training_data_to_folder(
            output_folder, table_name, limit, page_size, search_limit
        )
    elif train_or_test == "test":
        d.save_test_data_to_folder(
            output_folder, table_name, limit, page_size, search_limit
        )


if __name__ == "__main__":
    cli()
