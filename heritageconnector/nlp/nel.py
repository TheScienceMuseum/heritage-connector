from typing import List
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import textdistance
from hc_nlp.constants import ORG_LEGAL_SUFFIXES
from heritageconnector.base.disambiguation import Classifier
from heritageconnector import logging

logger = logging.get_logger(__name__)


class NELFeatureGenerator:
    """
    Generate a feature matrix `X` and optional target vector `y` from a DataFrame (`data`) containing the following columns:
    - the entity mention (`ent_mention_col`)
    - the predicted entity type (`ent_type_col`)
    - the entity context: either the entire piece of text or the sentence containing it (`ent_context_col`)
    - the candidate title (`candidate_title_col`)
    - the candidate type (`candidate_type_col`). This doesn't need to be mapped to entity type as each is one-hot-encoded separately.
    - the candidate context, for example a description of the candidate (`candidate_context_col`)

    Other parameters are:
    - `sbert_model` (str, Optional): the name of a pretrained model or the path to a custom model according to `sentence-transformers` docs [https://github.com/UKPLab/sentence-transformers].
    - `suffix_list` (List[str], Optional): a feature is generated which is the fuzzywuzzy token_sort_ratio between the entity context and candidate context with suffixes removed. By default a list of
        Organisation suffixes from `hc_nlp.constants.ORG_LEGAL_SUFFIXES` is used.

    From this feature matrix and optional target vector any general classifier e.g. from scikit-learn can be trained and used.

    The features generated here are broadly from Klie, Jan-Christoph, Richard Eckart de Castilho, and Iryna Gurevych. "From zero to hero: Human-in-the-loop entity linking in low resource domains." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        ent_mention_col: str,
        ent_type_col: str,
        ent_context_col: str,
        candidate_title_col: str,
        candidate_type_col: str,
        candidate_context_col: str,
        sbert_model: str = "stsb-distilbert-base",
        suffix_list: List[str] = ORG_LEGAL_SUFFIXES,
    ):
        self.data = data

        # TODO: do lowercase transformation here to make all methods case-insensitive

        self.ent_mention_col = self.data[ent_mention_col]
        self.ent_type_col = self.data[ent_type_col]
        self.ent_context_col = self.data[ent_context_col]
        self.candidate_title_col = self.data[candidate_title_col]
        self.candidate_type_col = self.data[candidate_type_col]
        self.candidate_context_col = self.data[candidate_context_col]

        self.suffix_list = suffix_list

        self.n_records = self.data.shape[0]

        self.ent_type_encoder = OneHotEncoder().fit(
            np.sort(self.ent_type_col.unique()).reshape(-1, 1)
        )
        self.candidate_type_encoder = OneHotEncoder().fit(
            np.sort(self.candidate_type_col.unique()).reshape(-1, 1)
        )

        self.bert_model = SentenceTransformer(sbert_model)

    @staticmethod
    def _remove_suffixes(text: str, suffix_list: List[str]) -> str:
        """
        Returns lowercased version of text with any of the suffixes in suffix_list removed. Case-insensitive.
        """
        mod_text = text[:-1].lower() if text[-1] == "." else text.lower()

        for suffix in suffix_list:
            if mod_text.endswith(suffix.lower()):
                mod_text = mod_text.rstrip(suffix.lower()).strip()

                break

        return mod_text

    def _apply_string_sim_method(
        self,
        method,
        col_a: pd.Series,
        col_b: pd.Series,
        token_wise: bool,
        denominator: int = 1,
    ) -> np.ndarray:
        """
        Params:
        - token_wise (bool): if True, split each string by spaces (`method` is passed two sequences rather than two strings)
        """
        if token_wise:
            return np.array(
                [
                    [
                        method(col_a.iloc[idx].split(), col_b.iloc[idx].split())
                        / denominator
                    ]
                    if all([pd.notnull(col_a.iloc[idx]), pd.notnull(col_b.iloc[idx])])
                    else [0]
                    for idx in range(self.n_records)
                ]
            )
        else:
            return np.array(
                [
                    [method(col_a.iloc[idx], col_b.iloc[idx]) / denominator]
                    if all([pd.notnull(col_a.iloc[idx]), pd.notnull(col_b.iloc[idx])])
                    else [0]
                    for idx in range(self.n_records)
                ]
            )

    def _generate_similarity_fuzz_sort(
        self, col_a: pd.Series, col_b: pd.Series, **kwargs
    ) -> np.ndarray:
        return self._apply_string_sim_method(
            fuzz.token_sort_ratio, col_a, col_b, denominator=100, token_wise=False
        )

    def _generate_similarity_levenshtein(
        self, col_a: pd.Series, col_b: pd.Series, **kwargs
    ) -> np.ndarray:
        return self._apply_string_sim_method(
            textdistance.levenshtein.normalized_similarity,
            col_a,
            col_b,
            token_wise=False,
        )

    def _generate_similarity_jarowinkler(
        self, col_a: pd.Series, col_b: pd.Series, **kwargs
    ) -> np.ndarray:
        return self._apply_string_sim_method(
            textdistance.jaro_winkler.normalized_similarity,
            col_a,
            col_b,
            token_wise=False,
        )

    def _generate_similarity_jaccard(
        self, col_a: pd.Series, col_b: pd.Series, **kwargs
    ) -> np.ndarray:
        return self._apply_string_sim_method(
            textdistance.jaccard.normalized_similarity, col_a, col_b, token_wise=True
        )

    def _generate_similarity_sorensen_dice(
        self, col_a: pd.Series, col_b: pd.Series, **kwargs
    ) -> np.ndarray:
        return self._apply_string_sim_method(
            textdistance.sorensen_dice.normalized_similarity,
            col_a,
            col_b,
            token_wise=True,
        )

    def _generate_ml_similarity_fuzz_sort_ignore_suffixes(self, **kwargs) -> np.ndarray:

        if "string_sim_metric" in kwargs:
            return np.array(
                [
                    [
                        kwargs["string_sim_metric"](
                            self.ent_mention_col.iloc[idx],
                            self.candidate_title_col.iloc[idx],
                        )
                        / 100
                    ]
                    for idx in range(self.n_records)
                ]
            )

        else:
            return np.array(
                [
                    [
                        fuzz.token_sort_ratio(
                            self._remove_suffixes(
                                self.ent_mention_col.iloc[idx], self.suffix_list
                            ),
                            self._remove_suffixes(
                                self.candidate_title_col.iloc[idx], self.suffix_list
                            ),
                        )
                        / 100
                    ]
                    for idx in range(self.n_records)
                ]
            )

    def _generate_label_in_mention(self, **kwargs) -> np.ndarray:
        return np.array(
            [
                [
                    float(
                        self.candidate_title_col.iloc[idx].lower()
                        in self.ent_mention_col.iloc[idx].lower()
                    )
                ]
                for idx in range(self.n_records)
            ]
        )

    def _generate_mention_in_label(self, **kwargs) -> np.ndarray:
        return np.array(
            [
                [
                    float(
                        self.ent_mention_col.iloc[idx].lower()
                        in self.candidate_title_col.iloc[idx].lower()
                    )
                ]
                for idx in range(self.n_records)
            ]
        )

    def _generate_type_features(self, **kwargs) -> np.ndarray:
        return np.concatenate(
            (
                self.ent_type_encoder.transform(
                    self.ent_type_col.values.reshape(-1, 1)
                ).toarray(),
                self.candidate_type_encoder.transform(
                    self.candidate_type_col.values.reshape(-1, 1)
                ).toarray(),
            ),
            axis=1,
        )

    def _generate_sentence_bert_cosdist_mention_label(self, **kwargs) -> np.ndarray:
        missing_similarity_value = kwargs.get("missing_sim_value", 0.5)

        logger.info("Calculating entity context sBERT embeddings... (1/2)")
        ent_descriptions = self.ent_context_col.astype(str).tolist()
        ent_description_embs = self.bert_model.encode(
            ent_descriptions, convert_to_tensor=True, show_progress_bar=True
        )

        logger.info("Calculating candidate context sBERT embeddings... (2/2)")
        candidate_descriptions = self.candidate_context_col.astype(str).tolist()
        candidate_description_embs = self.bert_model.encode(
            candidate_descriptions, convert_to_tensor=True, show_progress_bar=True
        )

        cosine_scores = util.pytorch_cos_sim(
            ent_description_embs, candidate_description_embs
        )

        sim_scores = np.array(
            [float(cosine_scores[i][i]) for i in range(len(ent_description_embs))]
        )

        # replace scores for missing values in either ent or candidate column with a fixed value
        nan_value_locs = [
            i
            for i in range(len(ent_descriptions))
            if (ent_descriptions[i] == "nan") or (candidate_descriptions[i] == "nan")
        ]
        np.put(sim_scores, nan_value_locs, [missing_similarity_value])

        return sim_scores.reshape(-1, 1)

    def get_feature_matrix(self) -> np.ndarray:
        """
        Calculate and retrieve a feature matrix (`X`) based on the columns provided in the DataFrame.

        Returns:
            np.ndarray: X
        """
        feats = (
            self._generate_similarity_fuzz_sort(
                self.ent_mention_col, self.candidate_title_col
            ),
            self._generate_similarity_levenshtein(
                self.ent_mention_col, self.candidate_title_col
            ),
            self._generate_similarity_jarowinkler(
                self.ent_mention_col, self.candidate_title_col
            ),
            self._generate_ml_similarity_fuzz_sort_ignore_suffixes(),
            self._generate_similarity_jarowinkler(
                self.ent_context_col, self.candidate_context_col
            ),
            self._generate_similarity_jaccard(
                self.ent_context_col, self.candidate_context_col
            ),
            self._generate_similarity_sorensen_dice(
                self.ent_context_col, self.candidate_context_col
            ),
            self._generate_label_in_mention(),
            self._generate_mention_in_label(),
            self._generate_type_features(),
            self._generate_sentence_bert_cosdist_mention_label(missing_sim_value=0.5),
        )

        feature_matrix = np.concatenate(feats, axis=1)

        return feature_matrix

    def get_target_vector(self, target_values_column: str) -> List[float]:
        """
        Get target vector (`y`) from data given a column containing the labels. If the `get_links_data_for_review` function of `heritageconnector.datastore.NERLoader` has been used
        then this should be the `link_correct` column.

        NOTE: to use this function, the DataFrame provided to the class instance should already be filtered so that there are no NaN values in `target_values_column`. Failure to do this
        will result in this function raising a ValueError.

        Args:
            target_values_column (str): column name of DataFrame containing target values. Can either be [True, False] or [1, 0].

        Returns:
            List[float]: Values are either 1 (positive) or 0 (negative).
        """

        if target_values_column not in self.data.columns:
            raise KeyError(
                f"Column `{target_values_column}`` is not a column in the DataFrame that has been provided to this class instance (which you can access with `instance.data`)."
            )

        if self.data[target_values_column].isna().sum() > 0:
            raise ValueError(
                f"Column `{target_values_column}` contains some NaN values. Please fill or remove these and rerun."
            )

        if set(self.data[target_values_column]) not in ({1, 0}, {True, False}):
            raise ValueError(
                f"Column `{target_values_column}` contains values other than [1, 0] or [True, False]. Please fix these and rerun."
            )

        return list(1 * (self.data[target_values_column].values))

    def get_feature_names(self) -> List[str]:
        """
        Get the name of each feature in X. Each element of the list corresponds to a column of X. This means that the list returned
        will contain duplicate values if a feature spans more than one column of X.

        Returns:
            List[str]: features in X
        """

        return (
            [
                "fuzz_sort similarity (mention-title)",
                "levenshtein similarity (mention-title)",
                "jaro-winkler similarity (mention-title)",
                "fuzz_sort similarity, ignoring suffixes (mention-title)",
                "jaro-winkler similarity (context-context)",
                "jaccard similarity (context-context)",
                "sorensen-dice similarity (context-context)",
                "label is in mention",
                "mention is in label",
            ]
            + [f"entity type ({t})" for t in list(self.ent_type_encoder.categories_)[0]]
            + [
                f"candidate type ({t})"
                for t in list(self.candidate_type_encoder.categories_)[0]
            ]
            + ["sBERT embedding cosine similarity (context-context)"]
        )
