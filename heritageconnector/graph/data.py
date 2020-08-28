import pandas as pd
import numpy as np
import os
import errno
from typing import Tuple, Union


def load_from_ntriples(path: str) -> np.ndarray:
    """
    Load RDF from ntriples format into a numpy array.

    Args:
        path (str): path to an ntriples (.nt) file

    Returns:
        np.ndarray
    """

    df = pd.read_csv(
        path, sep=" ", header=None, names=None, dtype=str, usecols=[0, 1, 2]
    )

    return df.to_numpy(dtype=str, copy=True)


def load_from_csv(path: str, sep: str = "\t", header: int = None) -> np.ndarray:
    """
    Load RDF from CSV format into a numpy array, dropping duplicate triples.

    For large graphs or those with mixed data types (e.g. numbers), use `load_from_triples` instead.

    Args:
        path (str): path to a CSV (.csv/.tsv) file
        sep (str, optional): Separator for the CSV file. Defaults to "\t".
        header (int, optional): Number of the header row (same as the pandas argument). Default to None. 

    Returns:
        np.ndarray
    """

    df = pd.read_csv(path, sep=sep, header=header, names=None, dtype=str)
    df = df.drop_duplicates()

    return df.to_numpy(dtype=str, copy=True)


def train_test_split(
    X: np.ndarray,
    test_size: Union[int, float],
    seed: int = 42,
    test_predicates: list = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate train-test split from an array of triples X. Test set is generated so that there are no unseen entities or relations, i.e. every entity 
    and relation in the test set exists in training. 

    Fork of `ampligraph.evaluation.train_test_split_no_unseen` to avoid a tensorflow dependency.

    Args:
        X (np.ndarray): array of triples, created using `load_from_csv` or `load_from_ntriples`
        test_size (int or float): number of triples in test set. If integer >= 1 is passed this is the length of the test set. If float < 1 is passed this is a
            proportion of X, e.g. 0.1 -> 10% of X.
        seed (int, optional): random seed. Defaults to 42.
        test_predicates (list, optional): A specific list of predicates to include in the test set. Useful for targeting model performance towards a 
            certain 'type' of triple. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: train_set, test_set
    """

    if isinstance(test_size, float) and test_size >= 1:
        test_size = int(test_size)
    elif isinstance(test_size, float) and test_size < 1:
        test_size = int(len(X) * test_size)

    rnd = np.random.RandomState(seed)

    subs, subs_cnt = np.unique(X[:, 0], return_counts=True)
    objs, objs_cnt = np.unique(X[:, 2], return_counts=True)
    rels, rels_cnt = np.unique(X[:, 1], return_counts=True)
    dict_subs = dict(zip(subs, subs_cnt))
    dict_objs = dict(zip(objs, objs_cnt))
    dict_rels = dict(zip(rels, rels_cnt))

    idx_test = np.array([], dtype=int)

    loop_count = 0
    tolerance = len(X) * 10
    # Set the indices of test set triples. If filtered, reduce candidate triples to certain predicate types.
    if test_predicates:
        test_triples_idx = np.where(np.isin(X[:, 1], test_predicates))[0]
    else:
        test_triples_idx = np.arange(len(X))

    while idx_test.shape[0] < test_size:
        i = rnd.choice(test_triples_idx)
        if dict_subs[X[i, 0]] > 1 and dict_objs[X[i, 2]] > 1 and dict_rels[X[i, 1]] > 1:
            dict_subs[X[i, 0]] -= 1
            dict_objs[X[i, 2]] -= 1
            dict_rels[X[i, 1]] -= 1

            idx_test = np.unique(np.append(idx_test, i))

        loop_count += 1

        # in case can't find solution
        if loop_count == tolerance:
            raise Exception(
                "Cannot create a test split of the desired size. "
                "Some entities will not occur in both training and test set. "
                "change seed values, remove filter on test predicates or "
                "set test_size to a smaller value."
            )

    idx = np.arange(len(X))
    idx_train = np.setdiff1d(idx, idx_test)

    return X[idx_train, :], X[idx_test, :]


def numpy_array_to_csv(
    array: np.ndarray, output_folder: str, file_name: str, sep: str = "\t"
):
    """
    Export RDF array to CSV with chosen separator (default \t).

    Args:
        array (np.ndarray): array containing triples
        output_folder (str): folder to output CSV file to
        file_name (str): name of exported CSV file
        sep (str, optional): CSV separator. Defaults to "\t".
    """

    if not os.path.exists(output_folder):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), output_folder)

    output_path = os.path.join(output_folder, file_name)

    df = pd.DataFrame(array)
    df.to_csv(output_path, sep=sep, index=False, header=False)

    # np.savetxt(output_path, array, delimiter=sep)
