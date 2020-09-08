import sys

sys.path.append("..")

import os
import numpy as np
import csv
import click
from heritageconnector.disambiguation.pipelines import build_training_data


@click.command()
@click.option("--output_folder", "-o", type=click.Path(exists=True))
@click.option("--table_name", "-t", type=str)
@click.option("--limit", "-l", type=int, default=None)
def make_training_data(
    output_folder, table_name, limit, page_size=250, search_limit=20
):
    X, y, pid_labels, id_pairs = build_training_data(
        table_name, page_size=page_size, search_limit=search_limit, limit=limit
    )

    np.save(os.path.join(output_folder, "X.npy"), X)
    np.save(os.path.join(output_folder, "y.npy"), y)

    with open(os.path.join(output_folder, "pids.txt"), "w") as f:
        f.write("\n".join(pid_labels))

    with open(os.path.join(output_folder, "ids.txt"), "w") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerows(id_pairs)


if __name__ == "__main__":
    make_training_data()
    # make_training_data(".", "PERSON", None)
