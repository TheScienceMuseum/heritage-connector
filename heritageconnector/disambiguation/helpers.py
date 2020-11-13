import os
from typing import Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve


def load_training_data(
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, list]:
    """
    Loads X, y, column and row labels from the folder specified by data_path. Also adds `is_type` column to pairs dataframe.

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

    pairs = pd.read_csv(
        os.path.join(data_path, "ids.txt"),
        header=None,
        delimiter="\t",
        names=["internal_id", "wikidata_id"],
    )
    pids = pd.read_csv(
        os.path.join(data_path, "pids.txt"), header=None, names=["column"]
    )["column"].tolist()

    type_idx = pids.index("P31")
    pairs["is_type"] = X[:, type_idx] > 0.01

    if os.path.exists(os.path.join(data_path, "y.npy")):
        y = np.load(os.path.join(data_path, "y.npy"))

        return X, y, pairs, pids
    else:
        return X, pairs, pids


def plot_performance_curves(y_true, y_pred_proba):
    """
    Plots TPR/FPR and Precision-Recall curves
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    precision, recall, thresholds2 = precision_recall_curve(y_true, y_pred_proba)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

    # plot TPR, FPR against threshold
    ax1.plot(thresholds, fpr, label="FPR")
    ax1.plot(thresholds, tpr, label="TPR")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("threshold")
    ax1.set_ylabel("TPR/FPR")
    ax1.legend(loc="best")
    ax1.set_title("TPR & FPR over Threshold")
    sns.despine()

    # plot precision-recall curve
    ax2.plot(recall, precision)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title("Precision-Recall Curve")
    sns.despine()


def filter_top_ranked_results(
    pairs: pd.DataFrame,
    enforce_correct_type: bool = True,
    max_wikidata_links: Union[int, None] = 3,
) -> pd.DataFrame:
    """
    Takes the dataframe returned by the disambiguator, and filters the results down to:
    - whether the result is of the correct type (`enforce_correct_type`)
    - only internal items with a maximum number of Wikidata links (`max_wikidata_links`)

    Args:
        pairs (pd.DataFrame): Returned by disambiguator.predict_top_ranked_pairs. Columns internal_id, wikidata_id, y_pred, y_pred_proba, is_type
        enforce_correct_type (bool, optional): Defaults to True.
        max_wikidata_links (Union[int, None], optional): Less than or equal to. Set to None to disable this filtering step. Defaults to 3.

    Returns:
        pd.DataFrame: Filtered dataframe with same columns as input.
    """

    if set(pairs.columns.tolist()) != {
        "internal_id",
        "wikidata_id",
        "y_pred",
        "y_pred_proba",
        "is_type",
    }:
        raise ValueError(
            "Input dataframe does not contain correct columns (internal_id, wikidata_id, y_pred, y_pred_proba, is_type)."
            "Make sure it has been created by disambiguator.predict_top_ranked_pairs."
        )

    pairs_filtered = pairs.copy()

    if enforce_correct_type:
        pairs_filtered = pairs_filtered[pairs_filtered["is_type"] == True]  # noqa: E712

    if max_wikidata_links is not None:
        links_per_internal_id = pairs_filtered["internal_id"].value_counts()
        filtered_internal_ids = links_per_internal_id[
            links_per_internal_id <= max_wikidata_links
        ].index.tolist()
        pairs_filtered = pairs_filtered[
            pairs_filtered["internal_id"].isin(filtered_internal_ids)
        ]

    return pairs_filtered
