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
    if train_or_test == "train":
        make_training_data(output_folder, table_name, limit)
    elif train_or_test == "test":
        make_test_data(output_folder, table_name, limit)


def make_training_data(
    output_folder, table_name, limit, page_size=100, search_limit=20
):
    d = Disambiguator()
    X, y, pid_labels, id_pairs = d.build_training_data(
        True, table_name, page_size=page_size, search_limit=search_limit, limit=limit
    )

    np.save(os.path.join(output_folder, "X.npy"), X)
    np.save(os.path.join(output_folder, "y.npy"), y)

    with open(os.path.join(output_folder, "pids.txt"), "w") as f:
        f.write("\n".join(pid_labels))

    with open(os.path.join(output_folder, "ids.txt"), "w") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerows(id_pairs)


def make_test_data(output_folder, table_name, limit, page_size=100, search_limit=20):
    d = Disambiguator()
    X, pid_labels, id_pairs = d.build_training_data(
        False, table_name, page_size=page_size, search_limit=search_limit, limit=limit
    )

    np.save(os.path.join(output_folder, "X.npy"), X)

    with open(os.path.join(output_folder, "pids.txt"), "w") as f:
        f.write("\n".join(pid_labels))

    with open(os.path.join(output_folder, "ids.txt"), "w") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerows(id_pairs)


if __name__ == "__main__":
    cli()

    ### DEBUG: ###
    # make_training_data(
    #     "/Volumes/Kalyan_SSD/SMG/disambiguation/test", "PERSON", limit=1000, page_size=100, search_limit=20
    # )

    # make_test_data(
    #     ".", "PERSON", limit=500, page_size=100, search_limit=20
    # )
