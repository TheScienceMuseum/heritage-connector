import numpy as np
import pandas as pd
import os
from typing import Tuple


def load_training_data(
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, list]:
    """
    Loads X, y, column and row labels from the folder specified by data_path.

    Args:
        data_path (str): folder containing X.npy, y.npy, ids.txt and pids.txt created by heritageconnector.disambiguation.pipelines.build_training_data

    Raises:
        FileNotFoundError: if data_path does not exist

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame, list]: X, y, internal-wikidata pairs, pids
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No such directory: {data_path}")

    X = np.load(os.path.join(data_path, "X.npy"))
    y = np.load(os.path.join(data_path, "y.npy"))

    pairs = pd.read_csv(
        os.path.join(data_path, "ids.txt"),
        header=None,
        delimiter="\t",
        names=["internal_id", "wikidata_id"],
    )
    pids = pd.read_csv(
        os.path.join(data_path, "pids.txt"), header=None, names=["column"]
    )["column"].tolist()

    return X, y, pairs, pids
