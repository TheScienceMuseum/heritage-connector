from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import textdistance
from hc_nlp.constants import ORG_LEGAL_SUFFIXES
from heritageconnector.base.disambiguation import Classifier
from heritageconnector import logging

logger = logging.get_logger(__name__)


class NELFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate a feature matrix `X` and optional target vector `y` from a DataFrame (`data`) containing the following columns:
    - the entity mention (`ent_mention_col`)
    - the predicted entity type (`ent_type_col`)
    - the entity context: either the entire piece of text or the sentence containing it (`ent_context_col`)
    - the candidate title (`candidate_title_col`)
    - the candidate type (`candidate_type_col`). This doesn't need to be mapped to entity type as each is one-hot-encoded separately.
    - the candidate context, for example a description of the candidate (`candidate_context_col`)

    Other parameters are:
    - `sbert_model_name_or_path` (str, Optional): the name of a pretrained model or the path to a custom model according to `sentence-transformers` docs [https://github.com/UKPLab/sentence-transformers].
    - `suffix_list` (List[str], Optional): a feature is generated which is the fuzzywuzzy token_sort_ratio between the entity context and candidate context with suffixes removed. By default a list of
        Organisation suffixes from `hc_nlp.constants.ORG_LEGAL_SUFFIXES` is used.

    From this feature matrix and optional target vector any general classifier e.g. from scikit-learn can be trained and used.

    The features generated here are broadly from Klie, Jan-Christoph, Richard Eckart de Castilho, and Iryna Gurevych. "From zero to hero: Human-in-the-loop entity linking in low resource domains." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020.
    """

    def __init__(
        self,
        sbert_model_name_or_path: str = "stsb-distilbert-base",
        suffix_list: List[str] = ORG_LEGAL_SUFFIXES,
    ):
        self.suffix_list = suffix_list
        self.sbert_model_name_or_path = sbert_model_name_or_path

    def fit(
        self,
        data: pd.DataFrame,
        ent_mention_col: str,
        ent_type_col: str,
        ent_context_col: str,
        candidate_title_col: str,
        candidate_type_col: str,
        candidate_context_col: str,
    ):
        # fit one-hot-encoders: these are used for generating the entity type and candidate type features
        self.ent_type_encoder = OneHotEncoder().fit(
            np.sort(data[ent_type_col].unique()).reshape(-1, 1)
        )
        self.candidate_type_encoder = OneHotEncoder().fit(
            np.sort(data[candidate_type_col].unique()).reshape(-1, 1)
        )

        self.ent_mention_col = ent_mention_col
        self.ent_type_col = ent_type_col
        self.ent_context_col = ent_context_col
        self.candidate_title_col = candidate_title_col
        self.candidate_type_col = candidate_type_col
        self.candidate_context_col = candidate_context_col

        return self

    # TODO: remove kwargs here. Column names should be set in `fit` above
    def transform(
        self,
        data: pd.DataFrame,
    ):
        self.sbert_model = SentenceTransformer(self.sbert_model_name_or_path)

        # TODO: create lowercase copy of data here
        X = self._calculate_features(
            data,
            self.ent_mention_col,
            self.ent_type_col,
            self.ent_context_col,
            self.candidate_title_col,
            self.candidate_type_col,
            self.candidate_context_col,
        )

        return X

    def fit_transform(
        self,
        data: pd.DataFrame,
        y: List[float],
        ent_mention_col: str,
        ent_type_col: str,
        ent_context_col: str,
        candidate_title_col: str,
        candidate_type_col: str,
        candidate_context_col: str,
    ):
        return self.fit(
            data,
            ent_mention_col=ent_mention_col,
            ent_type_col=ent_type_col,
            ent_context_col=ent_context_col,
            candidate_title_col=candidate_title_col,
            candidate_type_col=candidate_type_col,
            candidate_context_col=candidate_context_col,
        ).transform(
            data,
        )

    def get_feature_names(self) -> List[str]:
        """
        Get the name of each feature in X. Each element of the list corresponds to a column of X. This means that the list returned
        will contain duplicate values if a feature spans more than one column of X.

        Returns:
            List[str]: features in X
        """

        # TODO: rewrite for scikit-learn ecosystem

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
            # + ["sBERT embedding cosine similarity (mention-title)"]
            + ["sBERT embedding cosine similarity (context-context)"]
        )

    def _calculate_features(
        self,
        data: pd.DataFrame,
        ent_mention_col: str,
        ent_type_col: str,
        ent_context_col: str,
        candidate_title_col: str,
        candidate_type_col: str,
        candidate_context_col: str,
    ) -> np.ndarray:
        """
        Calculate and retrieve a feature matrix (`X`) based on the columns provided in the DataFrame.

        Returns:
            np.ndarray: X
        """

        feats = (
            self._generate_similarity_fuzz_sort(
                data[ent_mention_col], data[candidate_title_col]
            ),
            self._generate_similarity_levenshtein(
                data[ent_mention_col], data[candidate_title_col]
            ),
            self._generate_similarity_jarowinkler(
                data[ent_mention_col], data[candidate_title_col]
            ),
            self._generate_similarity_fuzz_sort_ignore_suffixes(
                data[ent_mention_col], data[candidate_title_col]
            ),
            self._generate_similarity_jarowinkler(
                data[ent_context_col], data[candidate_context_col]
            ),
            self._generate_similarity_jaccard(
                data[ent_context_col], data[candidate_context_col]
            ),
            self._generate_similarity_sorensen_dice(
                data[ent_context_col], data[candidate_context_col]
            ),
            self._generate_col_a_in_col_b(
                data[ent_mention_col], data[candidate_title_col]
            ),
            self._generate_col_a_in_col_b(
                data[candidate_title_col], data[ent_mention_col]
            ),
            self._generate_type_features(data[ent_type_col], data[candidate_type_col]),
            # TODO: make missing_sim_value a parameter of __init__
            # self._generate_sentence_bert_cosdist(
            #     data[ent_mention_col],
            #     data[candidate_title_col],
            #     missing_sim_value=0.5,
            # ),
            self._generate_sentence_bert_cosdist(
                data[ent_context_col],
                data[candidate_context_col],
                missing_sim_value=0.5,
            ),
        )

        feature_matrix = np.concatenate(feats, axis=1)

        return feature_matrix

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

        n_records = len(col_a)

        if token_wise:
            return np.array(
                [
                    [
                        method(col_a.iloc[idx].split(), col_b.iloc[idx].split())
                        / denominator
                    ]
                    if all([pd.notnull(col_a.iloc[idx]), pd.notnull(col_b.iloc[idx])])
                    else [0]
                    for idx in range(n_records)
                ]
            )
        else:
            return np.array(
                [
                    [method(col_a.iloc[idx], col_b.iloc[idx]) / denominator]
                    if all([pd.notnull(col_a.iloc[idx]), pd.notnull(col_b.iloc[idx])])
                    else [0]
                    for idx in range(n_records)
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

    def _generate_similarity_fuzz_sort_ignore_suffixes(
        self, col_a: pd.Series, col_b: pd.Series, **kwargs
    ) -> np.ndarray:
        n_records = len(col_a)

        if "string_sim_metric" in kwargs:
            return np.array(
                [
                    [
                        kwargs["string_sim_metric"](
                            col_a.iloc[idx],
                            col_b.iloc[idx],
                        )
                        / 100
                    ]
                    for idx in range(n_records)
                ]
            )

        else:
            return np.array(
                [
                    [
                        fuzz.token_sort_ratio(
                            self._remove_suffixes(col_a.iloc[idx], self.suffix_list),
                            self._remove_suffixes(col_b.iloc[idx], self.suffix_list),
                        )
                        / 100
                    ]
                    for idx in range(n_records)
                ]
            )

    def _generate_col_a_in_col_b(
        self, col_a: pd.Series, col_b: pd.Series, **kwargs
    ) -> np.ndarray:
        n_records = len(col_a)

        return np.array(
            [
                [float(col_a.iloc[idx].lower() in col_b.iloc[idx].lower())]
                for idx in range(n_records)
            ]
        )

    def _generate_type_features(
        self, ent_type_col: pd.Series, candidate_type_col: pd.Series, **kwargs
    ) -> np.ndarray:
        return np.concatenate(
            (
                self.ent_type_encoder.transform(
                    ent_type_col.values.reshape(-1, 1)
                ).toarray(),
                self.candidate_type_encoder.transform(
                    candidate_type_col.values.reshape(-1, 1)
                ).toarray(),
            ),
            axis=1,
        )

    def _generate_sentence_bert_cosdist(
        self, col_a: pd.Series, col_b: pd.Series, **kwargs
    ) -> np.ndarray:
        missing_similarity_value = kwargs.get("missing_sim_value", 0.5)

        logger.info("Calculating sBERT embeddings... (1/2)")
        descriptions_a = col_a.astype(str).tolist()
        description_embs_a = self.sbert_model.encode(
            descriptions_a, convert_to_tensor=True, show_progress_bar=True
        )

        logger.info("Calculating sBERT embeddings... (2/2)")
        descriptions_b = col_b.astype(str).tolist()
        description_embs_b = self.sbert_model.encode(
            descriptions_b, convert_to_tensor=True, show_progress_bar=True
        )

        cosine_scores = util.pytorch_cos_sim(description_embs_a, description_embs_b)

        sim_scores = np.array(
            [float(cosine_scores[i][i]) for i in range(len(description_embs_a))]
        )

        # replace scores for missing values in either ent or candidate column with a fixed value
        nan_value_locs = [
            i
            for i in range(len(descriptions_a))
            if (descriptions_a[i] == "nan") or (descriptions_b[i] == "nan")
        ]
        np.put(sim_scores, nan_value_locs, [missing_similarity_value])

        return sim_scores.reshape(-1, 1)
