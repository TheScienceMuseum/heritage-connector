from typing import List, Dict, Optional, Generator
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import textdistance
from hc_nlp.constants import ORG_LEGAL_SUFFIXES
from heritageconnector.base.disambiguation import Classifier
from heritageconnector import datastore
from heritageconnector.utils.generic import paginate_generator
from heritageconnector.config import config
from heritageconnector import logging
from elasticsearch import helpers
import json
import requests
import time
import math
from itertools import islice
from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

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
    - `sbert_model_name_or_path` (str, Optional): the name of a pretrained model or the path to a custom model according to `sentence-transformers` docs [https://github.com/UKPLab/sentence-transformers]. The model should be uncased (not case-sensitive).
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
        """
        Fit feature generator. This sets up the one-hot-encoders for entity and candidate type
        as well as the DataFrame columns for any data passed to the `NELFeatureGenerator.transform`
        method.

        Args:
            data (pd.DataFrame)
            ent_mention_col (str): column name of the entity mention
            ent_type_col (str): column name of the entity type
            ent_context_col (str): column name of the entity context
            candidate_title_col (str): column name of the candidate title
            candidate_type_col (str): column name of the candidate type
            candidate_context_col (str): column name of the candidate context

        Returns:
            self: trained `NELFeatureGenerator` instance
        """
        # Set dataframe columns. This is done here due to the constraint that the `transform` method of a scikit-learn
        # estimator must only have a `data` argument.
        self.ent_mention_col = ent_mention_col
        self.ent_type_col = ent_type_col
        self.ent_context_col = ent_context_col
        self.candidate_title_col = candidate_title_col
        self.candidate_type_col = candidate_type_col
        self.candidate_context_col = candidate_context_col

        self.all_cols = {
            "ent_mention_col": self.ent_mention_col,
            "ent_context_col": self.ent_context_col,
            "ent_type_col": self.ent_type_col,
            "candidate_title_col": self.candidate_title_col,
            "candidate_context_col": self.candidate_context_col,
            "candidate_type_col": self.candidate_type_col,
        }

        # Fit one-hot-encoders: these are used for generating the entity type and candidate type features
        self.ent_type_encoder = OneHotEncoder().fit(
            np.sort(data[ent_type_col].unique()).reshape(-1, 1)
        )
        self.candidate_type_encoder = OneHotEncoder().fit(
            np.sort(data[candidate_type_col].unique()).reshape(-1, 1)
        )

        return self

    def transform(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Transform data. All transforms are done on a lowercased version of the data.

        Args:
            data (pd.DataFrame)

        Raises:
            ValueError: if any of the required columns are not in `data`

        Returns:
            X (np.ndarray): feature vector. Columns are features, rows correspond to rows in `data`.
        """

        missing_columns = len(set(self.all_cols.values()) - set(data.columns))
        if missing_columns > 0:
            raise ValueError(f"Columns {missing_columns} are not in the provided data.")

        self.sbert_model = SentenceTransformer(self.sbert_model_name_or_path)

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
        """
        Fit the transformer and then return the transformed data.
        """
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

    @property
    def column_names(self) -> Dict[str, str]:
        """
        Return a dictionary of `{column_role: column name, ...}`
        """

        if hasattr(self, "all_cols"):
            return self.all_cols
        else:
            raise Exception(
                "Transformer has not been fit yet so there are no column names. Call `NELFeatureGenerator.fit` first."
            )

    @property
    def feature_names(self) -> List[str]:
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

        _ent_mention_col = data[ent_mention_col].astype(str)
        _ent_context_col = data[ent_context_col].astype(str)
        _candidate_title_col = data[candidate_title_col].astype(str)
        _candidate_context_col = data[candidate_context_col].astype(str)

        feats = (
            self._generate_similarity_fuzz_sort(_ent_mention_col, _candidate_title_col),
            self._generate_similarity_levenshtein(
                _ent_mention_col, _candidate_title_col
            ),
            self._generate_similarity_jarowinkler(
                _ent_mention_col, _candidate_title_col
            ),
            self._generate_similarity_fuzz_sort_ignore_suffixes(
                _ent_mention_col, _candidate_title_col
            ),
            self._generate_similarity_jarowinkler(
                _ent_context_col, _candidate_context_col
            ),
            self._generate_similarity_jaccard(_ent_context_col, _candidate_context_col),
            self._generate_similarity_sorensen_dice(
                _ent_context_col, _candidate_context_col
            ),
            self._generate_col_a_in_col_b(_ent_mention_col, _candidate_title_col),
            self._generate_col_a_in_col_b(_candidate_title_col, _ent_mention_col),
            self._generate_type_features(data[ent_type_col], data[candidate_type_col]),
            self._generate_sentence_bert_cosdist(
                _ent_context_col,
                _candidate_context_col,
                # TODO: make missing_sim_value an argument to __init__?
                missing_sim_value=0.5,
            ),
        )

        feature_matrix = np.concatenate(feats, axis=1)

        return feature_matrix

    @staticmethod
    def _remove_suffixes(text: str, suffix_list: List[str]) -> str:
        """
        Returns text with any of the suffixes in suffix_list removed. Detection of suffixes is case-insensitive.
        """
        mod_text = text[:-1] if text[-1] == "." else text

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
            [[float(col_a.iloc[idx] in col_b.iloc[idx])] for idx in range(n_records)]
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
        """
        Embeddings are calculated on unique values of each column for efficiency.
        For any comparisons whether either the value of `col_a` or `col_b` is NaN,
        the value of optional kwarg `missing_sim_value` is used instead of the
        cosine similarity between "nan" and a string.
        """
        missing_similarity_value = kwargs.get("missing_sim_value", 0.5)

        logger.debug("Calculating sBERT embeddings... (1/2)")
        descriptions_a = col_a.astype(str).tolist()
        descriptions_a_unique_vals, descriptions_a_unique_indices = np.unique(
            descriptions_a, return_inverse=True
        )
        description_embs_a_unique = self.sbert_model.encode(
            descriptions_a_unique_vals, convert_to_tensor=True, show_progress_bar=False
        )
        description_embs_a = description_embs_a_unique[descriptions_a_unique_indices]

        logger.debug("Calculating sBERT embeddings... (2/2)")
        descriptions_b = col_b.astype(str).tolist()
        descriptions_b_unique_vals, descriptions_b_unique_indices = np.unique(
            descriptions_b, return_inverse=True
        )
        description_embs_b_unique = self.sbert_model.encode(
            descriptions_b_unique_vals, convert_to_tensor=True, show_progress_bar=False
        )
        description_embs_b = description_embs_b_unique[descriptions_b_unique_indices]

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


def get_target_values_from_review_data(data: pd.DataFrame, target_values_column: str):
    if target_values_column not in data.columns:
        raise KeyError(
            f"Column `{target_values_column}`` is not a column in the DataFrame that has been provided to this class instance (which you can access with `instance.data`)."
        )

    if data[target_values_column].isna().sum() > 0:
        raise ValueError(
            f"Column `{target_values_column}` contains some NaN values. Please fill or remove these and rerun."
        )

    if set(data[target_values_column]) not in ({1, 0}, {True, False}):
        raise ValueError(
            f"Column `{target_values_column}` contains values other than [1, 0] or [True, False]. Please fix these and rerun."
        )

    return list(1 * (data[target_values_column].values))


class BLINKServiceWrapper:
    """
    Wrapper around a BLINK service (REST API) such as the one at https://github.com/TheScienceMuseum/BLINK.
    """

    def __init__(
        self,
        blink_endpoint: str,
        description_field: str,
        entity_fields: List[str],
        wiki_link_threshold: Optional[float] = 0,
    ):
        """Initialise an instance of BLINKServiceWrapper.

        Args:
            blink_endpoint (str): API endpoint. E.g. 'http://localhost:8000/blink/multiple'
        """

        self.endpoint = blink_endpoint
        self.headers = {"Content-Type": "application/json"}

        self.description_field = description_field
        self.entity_fields = entity_fields

        self.link_threshold = wiki_link_threshold

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def make_blink_request(self, query: dict, silent: bool = False) -> dict:
        """Make a request to BLINK.

        See https://github.com/TheScienceMuseum/BLINK#using-the-rest-api for request and response formats.

        Args:
            query (dict): BLINK query body in Python dict format.
            silent (bool, optional): Whether to suppress logs (of the number of items processed and the time taken). Defaults to False.

        Returns:
            dict: BLINK response
        """
        request_len = len(query["items"])
        start = time.time()
        response_json = requests.request(
            "POST", self.endpoint, headers=self.headers, data=json.dumps(query)
        ).json()
        end = time.time()
        response_time = end - start
        response_rate = response_time / request_len

        if not silent:
            logger.info(
                f"{request_len} items processed in {round(response_time, 1)} seconds ({response_rate} seconds/item)"
            )

        return response_json

    def _get_unlinked_entities_generator(
        self, es_index: str, entity_fields: List[str]
    ) -> Generator[dict, None, None]:
        """Get generator which yields documents from an Elasticsearch index with at least one of the fields in `entity_fields`.

        Args:
            es_index (str): elasticsearch index
            entity_fields (List[str]): e.g. ["graph.@hc:entityPERSON.@value", "graph.@hc:entityORG.@value"]

        Yields:
            Generator[dict, None, None]: [description]
        """

        if not all([i.endswith("@value") for i in entity_fields]):
            logger.warning(
                "Some of the entity fields provided look like they're not valid JSON-LD (don't end in '@value')."
            )

        fields_exist = [{"exists": {"field": field}} for field in entity_fields]

        es_query = {"query": {"bool": {"should": fields_exist}}}

        doc_generator = helpers.scan(
            client=datastore.es,
            index=es_index,
            query=es_query,
            scroll="5h",
            size=500,
            # preserve_order=True,
        )

        return doc_generator

    def _covert_doc_to_blink_query_format(
        self, doc: dict, description_field: str, entity_fields: List[str]
    ) -> List[dict]:
        # get description and entity mentions
        uri = doc["_id"]
        description = datastore._get_dict_field_from_dot_notation(
            doc, description_field
        )
        ent_mentions = []

        for field_name in entity_fields:
            field_val = datastore._get_dict_field_from_dot_notation(doc, field_name)

            if isinstance(field_val, str) and field_val != "":
                ent_mentions.append((field_val, field_name))
            elif isinstance(field_val, list):
                ent_mentions += [(v, field_name) for v in field_val]

        blink_request_items = []

        for mention, label in ent_mentions:
            # there are edge cases in which the mention may not be in the description, because NER was run on a cleaned
            # version of the description
            if mention in description:
                # only the first mention is highlighted, if the entity mention occurs more than once in the description
                mod_desc = description.replace(
                    mention, f"[ENT_START]{mention}[ENT_END]", 1
                )
                blink_request_items.append(
                    {
                        "id": uri,
                        "text": mod_desc,
                        "metadata": {"mention": mention, "label": label},
                    }
                )

        return blink_request_items

    def _convert_page_of_docs_to_blink_query_format(
        self, page_of_docs: List[dict], description_field: str, entity_fields: List[str]
    ) -> List[dict]:
        output = []

        for doc in page_of_docs:
            output += self._covert_doc_to_blink_query_format(
                doc, description_field, entity_fields
            )

        return output

    def _write_items_to_jsonl_file(self, data: List[dict], file_path: str):
        """Append the items in `data` to the .jsonl file specified by `file_path`."""
        with open(file_path, "a") as f:
            for dict_obj in data:
                if len(dict_obj["links"]) > 0:
                    f.write(json.dumps(dict_obj))
                    f.write("\n")

    def process_unlinked_entity_mentions(
        self,
        es_index: str,
        output_path: str,
        page_size: int,
        limit: Optional[int] = None,
    ):
        # get generator for unlinked mentions
        doc_generator = self._get_unlinked_entities_generator(
            es_index, self.entity_fields
        )
        if limit is not None:
            doc_generator = islice(doc_generator, limit)

        doc_generator_paginated = paginate_generator(doc_generator, page_size)

        # no_pages = math.ceil(limit / page_size) if limit is not None else None

        bar = tqdm(total=None)

        for idx, page in enumerate(doc_generator_paginated):
            try:
                items = self._convert_page_of_docs_to_blink_query_format(
                    page, self.description_field, self.entity_fields
                )

                # convert these to a format that can be used in the BLINK query
                body = {
                    "items": items,
                    "threshold": self.link_threshold,
                }

                # stream through BLINK
                response_items = self.make_blink_request(body, silent=True).get("items")

                # save to JSON
                self._write_items_to_jsonl_file(response_items, output_path)

                # increment progress bar with number of entity mentions
                bar.update(len(items))

            except RetryError:
                logger.warn(f"page {idx} failed")
                pass
