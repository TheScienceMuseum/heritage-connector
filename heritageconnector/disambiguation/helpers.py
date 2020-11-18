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
