from elasticsearch import helpers, Elasticsearch
from elasticsearch import exceptions as es_exceptions
import rdflib
from rdflib import Graph, Literal, URIRef
from rdflib.serializer import Serializer
import json
import os
from typing import Generator, List, Tuple, Optional, Union, Iterable, Callable
from tqdm.auto import tqdm
from itertools import islice
from heritageconnector.namespace import (
    XSD,
    FOAF,
    OWL,
    RDF,
    RDFS,
    PROV,
    SDO,
    SKOS,
    WD,
    WDT,
    HC,
    get_jsonld_context,
)
from heritageconnector.utils.generic import paginate_generator, flatten_list_of_lists
from heritageconnector.config import config
from heritageconnector.nlp.nel import (
    NELFeatureGenerator,
    get_target_values_from_review_data,
)
from heritageconnector import logging, errors, best_spacy_pipeline
import pandas as pd
import spacy
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

logger = logging.get_logger(__name__)

# Should we implement this as a persistance class esp. for connection pooling?
# https://elasticsearch-dsl.readthedocs.io/en/latest/persistence.html

if hasattr(config, "ELASTIC_SEARCH_CLUSTER"):
    es = Elasticsearch(
        [config.ELASTIC_SEARCH_CLUSTER],
        http_auth=(config.ELASTIC_SEARCH_USER, config.ELASTIC_SEARCH_PASSWORD),
    )
    logger.debug(
        f"Connected to Elasticsearch cluster at {config.ELASTIC_SEARCH_CLUSTER}"
    )
else:
    # use localhost
    es = Elasticsearch()
    logger.debug("Connected to Elasticsearch cluster on localhost")

index = config.ELASTIC_SEARCH_INDEX
es_config = {
    "chunk_size": int(config.ES_BULK_CHUNK_SIZE),
    "queue_size": int(config.ES_BULK_QUEUE_SIZE),
}

context = get_jsonld_context()


class RecordLoader:
    """
    Contains functions for loading JSON-LD formatted data into an Elasticsearch index.
    """

    def __init__(self, collection_name: str, field_mapping: dict):
        """
        Args:
            collection_name (str): name of collection to populate `doc['_source']['collection']` for all records.
            field_mapping (dict): field mapping config
        """
        self.collection_name = collection_name
        self.field_mapping = field_mapping
        self.mapping = field_mapping.mapping
        self.non_graph_predicates = field_mapping.non_graph_predicates

    def _get_table_mapping(self, table_name: str) -> dict:
        """
        Get table mapping from field_mapping config. Raises if table_name doesn't exist in the `field_mapping.mapping` dictionary.
        """

        if table_name in self.mapping:
            return self.mapping[table_name]
        else:
            raise KeyError(
                f"Table name '{table_name}' doesn't exist in the mapping dictionary in your field_mapping config."
            )

    def add_record(
        self, table_name: str, record: pd.Series, add_type: rdflib.URIRef = None
    ):
        """
        Create and store new record with a JSON-LD graph.

        Args:
            table_name (str): name of table in `field_mapping.mapping` (top-level key)
            record (pd.Series): row from tabular data to import. Must contain column 'URI' specifying the URI which uniquely
                identifies each record, and column names must match up to keys in `field_mapping.mapping[table_name]`.
            add_type (rdflib.URIRef, optional): URIRef to add as a value for RDF.type in the record. Defaults to None.
        """

        uri = str(record["URI"])

        table_mapping = self._get_table_mapping(table_name)
        data_fields = [
            k
            for k, v in table_mapping.items()
            if v.get("RDF") in self.non_graph_predicates
        ]

        data = self._serialize_to_json(table_name, record, data_fields)
        data["uri"] = uri
        jsonld = self._serialize_to_jsonld(
            table_name,
            uri,
            record,
            ignore_types=self.non_graph_predicates,
            add_type=add_type,
        )

        create(self.collection_name, table_name, data, jsonld)

    def add_records(
        self, table_name: str, records: pd.DataFrame, add_type: rdflib.URIRef = None
    ):
        """
        Create and store multiple new records with a JSON-LD graph. Uses Elasticsearch's bulk import method (https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html)
        which is faster than calling `add_record` on each record.

        Args:
            table_name (str): name of table in `field_mapping.mapping` (top-level key)
            records (pd.DataFrame): tabular data to import. Must contain column 'URI' specifying the URI which uniquely
                identifies each record, and column names must match up to keys in `field_mapping.mapping[table_name]`.
            add_type (rdflib.URIRef, optional): URIRef to add as a value for RDF.type in the record. Defaults to None.
        """

        generator = self._record_create_generator(table_name, records, add_type)
        es_bulk(generator, len(records))

    def _record_create_generator(
        self, table_name: str, records: pd.DataFrame, add_type: rdflib.URIRef
    ) -> Generator[dict, None, None]:
        """Yields JSON-LD for the `add_records` method, to add new documents in bulk to the index.."""

        table_mapping = self._get_table_mapping(table_name)

        data_fields = [
            k
            for k, v in table_mapping.items()
            if v.get("RDF") in self.non_graph_predicates
        ]

        for _, record in records.iterrows():
            uri = str(record["URI"])

            data = self._serialize_to_json(table_name, record, data_fields)
            jsonld = self._serialize_to_jsonld(
                table_name,
                uri,
                record,
                ignore_types=self.non_graph_predicates,
                add_type=add_type,
            )

            doc = {
                "_id": uri,
                "uri": uri,
                "collection": self.collection_name,
                "type": table_name,
                "data": data,
                "graph": jsonld,
            }

            yield doc

    def add_triples(
        self,
        records: pd.DataFrame,
        predicate: rdflib.URIRef,
        subject_col: str = "SUBJECT",
        object_col: str = "OBJECT",
        object_is_uri: bool = True,
        progress_bar: bool = True,
    ):
        """
        Add triples with RDF predicate and dataframe containing subject and object columns.
        Values in object_col can either be string or list. If list, a one subject to many
        objects relationship is assumed.

        All items in `subject_col` must be a URI. The `object_is_uri` parameter defines whether an attempt will be made to load
        objects in as URIs (True) or Literals (False).

        Args:
            records (pd.DataFrame): dataframe containing subject and object columns with respective column names being the values of `subject_col` and `object_col`.
            predicate (rdflib.URIRef): predicate to connect these triples.
            subject_col (str, optional): name of column containing subject values. Defaults to "SUBJECT".
            object_col (str, optional): name of column containing object values. Defaults to "OBJECT".
            object_is_uri (bool, optional): whether the object column contains URIs. If False, will be loaded in as literals.
            progress_bar (bool, optional): whether to show a progress bar for loading into Elasticsearch. Defaults to True.
        """

        generator = self._record_update_generator(
            records, predicate, subject_col, object_col, object_is_uri
        )
        es_bulk(
            generator, len(records[subject_col].unique()), progress_bar=progress_bar
        )

    def _record_update_generator(
        self,
        df: pd.DataFrame,
        predicate: rdflib.URIRef,
        subject_col: str = "SUBJECT",
        object_col: str = "OBJECT",
        object_is_uri: bool = True,
    ) -> Generator[dict, None, None]:
        """Yields JSON-LD docs for the `add_triples` method, to update existing records with new triples"""

        # ensure that subject_col is unique by converting object_col to aggregated lists, otherwise object
        # values will be overwritten when adding to graph
        df = df.copy().groupby(subject_col).agg(list).reset_index()
        df[object_col] = df[object_col].apply(flatten_list_of_lists)

        # if this assertion is not true than entities are likely to be overwritten as there are repeats of values
        # in `subject_col`
        assert len(df[subject_col].unique()) == len(df)

        for _, row in df.iterrows():
            g = Graph()
            object_str_to_rdflib = URIRef if object_is_uri else Literal

            if isinstance(row[object_col], str):
                g.add(
                    (
                        URIRef(row[subject_col]),
                        predicate,
                        object_str_to_rdflib(row[object_col]),
                    )
                )
            elif isinstance(row[object_col], list):
                [
                    g.add(
                        (URIRef(row[subject_col]), predicate, object_str_to_rdflib(v))
                    )
                    for v in row[object_col]
                ]

            jsonld_dict = json.loads(
                g.serialize(format="json-ld", context=context, indent=4)
            )
            _ = jsonld_dict.pop("@id")
            _ = jsonld_dict.pop("@context")

            body = {"graph": jsonld_dict}

            doc = {"_id": row[subject_col], "_op_type": "update", "doc": body}

            yield doc

    def _serialize_to_json(
        self, table_name: str, record: pd.Series, columns: list
    ) -> dict:
        """Return a JSON representation of data fields to exist outside of the graph."""

        table_mapping = self._get_table_mapping(table_name)

        data = {}

        for col in columns:
            if (
                "RDF" in table_mapping[col]
                and bool(record[col])
                and (str(record[col]).lower() != "nan")
            ):
                # TODO: these lines load description in as https://collection.sciencemuseumgroup.org.uk/objects/co__#<field_name> but for some reason they cause an Elasticsearch timeout
                # key = str(row['URI']) + "#" + col.lower()
                # data[key] = row[col]
                data.update({table_mapping[col]["RDF"]: record[col]})

        return data

    def _serialize_to_jsonld(
        self,
        table_name: str,
        uri: str,
        row: pd.Series,
        ignore_types: list,
        add_type: rdflib.term.URIRef = None,
    ) -> dict:
        """
        Returns a JSON-LD represention of a record

        Args:
            table_name (str): given name of the table being imported
            uri (str): URI of subject
            row (pd.Series): DataFrame row (record) to serialize
            ignore_types (list): predicates to ignore when importing
            add_type (rdflib.term.URIRef, optional): whether to add @type field with the table_name. If a value rather than
                a boolean is passed in, this will be added as the type for the table. Defaults to True.

        Raises:
            KeyError: if a column listed in `field_mapping.mapping[table_name]` is not in the provided data table

        Returns:
            dict: JSON_LD formatted document
        """

        g = Graph()
        record = URIRef(uri)

        g.add((record, SKOS.hasTopConcept, Literal(table_name)))

        # Add RDF:type
        # Need to check for isinstance otherwise this will fail silently during bulk load, causing the entire record to not load
        if (add_type is not None) and isinstance(add_type, rdflib.term.URIRef):
            g.add((record, RDF.type, add_type))

        table_mapping = self._get_table_mapping(table_name)

        keys = {
            k
            for k, v in table_mapping.items()
            if "RDF" in v and v.get("RDF") not in ignore_types
        }

        for col in keys:
            # this will trigger for the first row in the dataframe
            if col not in row.index:
                raise KeyError(f"column {col} not in data for table {table_name}")

            if bool(row[col]) and (str(row[col]).lower() != "nan"):
                if isinstance(row[col], list):
                    [
                        g.add((record, table_mapping[col]["RDF"], Literal(val)))
                        for val in row[col]
                        if str(val) != "nan"
                    ]
                elif isinstance(row[col], URIRef):
                    g.add((record, table_mapping[col]["RDF"], row[col]))

                else:
                    g.add((record, table_mapping[col]["RDF"], Literal(row[col])))

        json_ld_dict = json.loads(
            g.serialize(format="json-ld", context=context, indent=4).decode("utf-8")
        )

        # "'@graph': []" appears when there are no linked objects to the document, which breaks the RDF conversion.
        # There is also no @id field in the graph when this happens.
        json_ld_dict.pop("@graph", None)
        json_ld_dict["@id"] = uri

        return json_ld_dict


def create_index():
    """Delete the exiting ES index if it exists and create a new index and mappings"""

    logger.info("Wiping existing index: " + index)
    es.indices.delete(index=index, ignore=[400, 404])

    # setup any mappings etc.
    indexSettings = {"settings": {"number_of_shards": 1, "number_of_replicas": 0}}

    logger.info("Creating new index: " + index)
    es.indices.create(index=index, body=indexSettings)
    logger.info("..done ")


def es_bulk(action_generator, total_iterations=None, progress_bar=True):
    """Batch load a set of new records into ElasticSearch"""

    successes = 0
    errs = []

    bulk_iterator = helpers.parallel_bulk(
        client=es,
        index=index,
        actions=action_generator,
        chunk_size=es_config["chunk_size"],
        queue_size=es_config["queue_size"],
        raise_on_error=False,
    )

    if progress_bar:
        bulk_iterator = tqdm(bulk_iterator, total=total_iterations)

    for ok, action in bulk_iterator:
        if not ok:
            errs.append(action)
        successes += ok

    return successes, errs


def create(collection, record_type, data, jsonld):
    """Load a new record into ElasticSearch and return its id"""

    # create a ES doc
    doc = {
        "uri": data["uri"],
        "collection": collection,
        "type": record_type,
        "data": {i: data[i] for i in data if i != "uri"},
        "graph": json.loads(jsonld),
    }
    es_json = json.dumps(doc)

    # add JSON document to ES index
    response = es.index(index=index, id=data["uri"], body=es_json)

    return response


def update_graph(s_uri, p, o_uri):
    """Add a new RDF relationship to an an existing record"""

    # create graph containing just the new triple
    g = Graph()
    g.add((URIRef(s_uri), p, URIRef(o_uri)))

    # export triple as JSON-LD and remove ID, context
    jsonld_dict = json.loads(g.serialize(format="json-ld", context=context, indent=4))
    _ = jsonld_dict.pop("@id")
    _ = jsonld_dict.pop("@context")

    body = {"doc": {"graph": jsonld_dict}}

    es.update(index=index, id=s_uri, body=body, ignore=404)


def get_by_uri(uri: str) -> dict:
    """Return an existing ElasticSearch record. Raise ValueError if record with specified URI doesn't exist."""

    try:
        res = es.get(index=index, id=uri)
        return res
    except es_exceptions.TransportError:
        raise ValueError(f"No record with uri {uri} exists.")


def get_by_type(type, size=1000):
    """Return an list of matching ElasticSearch record"""

    res = es.search(index=index, body={"query": {"match": {"type": type}}}, size=size)
    return res["hits"]["hits"]


def get_graph(uri):
    """Return an the RDF graph for an ElasticSearch record"""

    record = get_by_uri(uri)
    if record:
        jsonld = json.dumps(record["_source"]["graph"])
        g = Graph().parse(data=jsonld, format="json-ld")

    return g


def get_graph_by_type(type):
    """Return an list of matching ElasticSearch record"""

    g = Graph()
    records = get_by_type(type)
    for record in records:
        jsonld = json.dumps(record["_source"]["graph"])
        g.parse(data=jsonld, format="json-ld")

    return g


def es_to_rdflib_graph(g=None, return_format=None):
    """
    Turns a dump of ES index into an RDF format. Returns an RDFlib graph object if no
    format is specified, else an object with the specified format which could be written
    to a file.
    """

    # get dump
    res = helpers.scan(
        client=es, index=index, query={"_source": "graph.*", "query": {"match_all": {}}}
    )
    total = es.count(index=index)["count"]

    # create graph
    if g is None:
        g = Graph()

        for item in tqdm(res, total=total):
            g += Graph().parse(
                data=json.dumps(item["_source"]["graph"]), format="json-ld"
            )
    else:
        logger.debug("Using existing graph")
        for item in tqdm(res, total=total):
            g.parse(data=json.dumps(item["_source"]["graph"]), format="json-ld")

    if return_format is None:
        return g
    else:
        return g.serialize(format=return_format)


class NERLoader:
    def __init__(
        self,
        record_loader: RecordLoader,
        source_es_index: str,
        source_description_field: str,
        target_es_index: str,
        target_title_field: str,
        target_description_field: str,
        target_type_field: str,
        target_alias_field: str = None,
        # batch_size: Optional[int] = 1024,
        entity_types: Iterable[str] = [
            "PERSON",
            "ORG",
            "NORP",
            "FAC",
            "LOC",
            "OBJECT",
            "LANGUAGE",
            "DATE",
            "EVENT",
        ],
        entity_types_to_link: Iterable[str] = [
            "PERSON",
            "ORG",
            "OBJECT",
        ],
        text_preprocess_func: Optional[Callable[[str], str]] = None,
        entity_markers: Iterable[str] = ("[[", "]]"),
    ):
        """
        Initialise instance of NERLoader.

        Args:
            record_loader (RecordLoader): instance of RecordLoader, with parameters suitable for the current Heritage Connector index.
            entity_types (List[str], optional): entity types to extract from the spaCy model.
            entity_types_to_link (List[str], optional): entity types to try to link to records. Filtered to only types that appear in `entity_types`.
        """

        self.record_loader = record_loader

        self.entity_types = set(entity_types)
        self.entity_types_to_link = set(entity_types_to_link).intersection(
            self.entity_types
        )

        self.source_index = source_es_index
        self.target_index = target_es_index

        self.source_fields = {"description": source_description_field}

        self.target_fields = {
            "title": target_title_field,
            "description": target_description_field,
            "type": target_type_field,
        }

        if target_alias_field is not None:
            self.target_fields.update(
                {
                    "alias": target_alias_field,
                }
            )

        self.text_preprocess_func = (
            text_preprocess_func if text_preprocess_func is not None else lambda x: x
        )

        if not all([isinstance(i, str) for i in entity_markers]) or not (
            len(entity_markers) == 2
        ):
            raise ValueError(
                "Parameter `entity_markers` must be an iterable (e.g. list or tuple) containing exactly two items, both of which are strings."
            )
        self.entity_markers = entity_markers

        self._entity_list = []

    @property
    def entity_list(self) -> List[dict]:
        """
        List of `{"item_uri": _, "ent_label": _, "ent_text": _,}` dictionaries, with one dictionary per entity.
        Each dictionary has an optional `{"link_candidates": {}}` member if link candidates have been retrieved
        for that entity.
        """
        return self._entity_list

    @property
    def entity_list_as_dataframe(self) -> pd.DataFrame:
        ent_df = pd.DataFrame(self._entity_list)

        if "link_candidates" not in ent_df.columns:
            return ent_df

        # Get DataFrame of entity mentions without link candidates, either because they have types that haven't requested
        # to be linked (link_candidates column isna()) or because no results were returned from the linking search
        # (link candidates column is empty list).
        ents_no_candidates_df = pd.concat(
            [
                ent_df[ent_df["link_candidates"].isna()],
                ent_df[ent_df.astype(str)["link_candidates"] == "[]"],
            ]
        )

        # Transform data with link candidates for easy review.
        ents_with_candidates_df = (
            ent_df.dropna(subset=["link_candidates"])
            .set_index(
                [
                    "item_uri",
                    "item_description",
                    "item_description_with_ent",
                    "ent_label",
                    "ent_text",
                    "ent_sentence",
                ]
            )["link_candidates"]
            .apply(pd.Series)
            .stack()
            .reset_index()
            # the level of level_n below is the length of the list used in `set_index above`
            .rename(columns={"level_6": "candidate_rank"})
        )

        candidate_cols = ents_with_candidates_df[0].apply(pd.Series)
        candidate_cols = candidate_cols.rename(
            columns={col: f"candidate_{col}" for col in candidate_cols}
        )

        ents_with_candidates_transformed_df = (
            pd.concat([ents_with_candidates_df, candidate_cols], axis=1)
            .drop(columns=[0])
            .fillna("")
        )
        ents_with_candidates_transformed_df["link_correct"] = ""

        cols_order = [
            "item_uri",
            "candidate_rank",
            "item_description_with_ent",
            "ent_label",
            "ent_text",
            "ent_sentence",
            "candidate_title",
            "candidate_type",
            "candidate_uri",
            "link_correct",
            "candidate_alias",
            "candidate_description",
            "item_description",
        ]
        other_cols = [
            col
            for col in ents_with_candidates_transformed_df.columns
            if col not in cols_order
        ]

        # Return the concatenation of the DataFrame with links and the DataFrame without links.
        df_linked_and_unlinked = pd.concat(
            [ents_no_candidates_df, ents_with_candidates_transformed_df]
        )

        df_linked_and_unlinked = df_linked_and_unlinked[cols_order + other_cols]
        # ents_with_candidates_transformed_df = ents_with_candidates_transformed_df[cols_order + other_cols]

        # for debugging: check all records have ended up in final dataframe
        assert set(df_linked_and_unlinked["item_uri"]) == set(
            [item["item_uri"] for item in self._entity_list]
        )

        return df_linked_and_unlinked

    def get_links_data_for_review(self) -> pd.DataFrame:
        """
        Returns a DataFrame of only entity mentions with a set of link candidates,
        with an entity-candidate pair on each line. Also creates a blank column for
        review results.
        """

        links_df = self.entity_list_as_dataframe.copy()
        links_df = links_df[~links_df["candidate_rank"].isnull()]

        return links_df

    def train_entity_linker(
        self,
        train_data: pd.DataFrame,
        ent_mention_col: str = "ent_text",
        ent_type_col: str = "ent_label",
        ent_context_col: str = "item_description",
        candidate_title_col: str = "candidate_title",
        candidate_type_col: str = "candidate_type",
        candidate_context_col: str = "candidate_description",
        target_col: str = "link_correct",
        sbert_model: Optional[str] = None,
        suffix_list: Optional[str] = None,
        linking_classifier: BaseEstimator = MLPClassifier,
        classifier_kwargs: dict = {"random_state": 42, "max_iter": 1000},
        random_seed: int = 42,
    ) -> BaseEstimator:
        """
        Train entity linking binary classifier using a DataFrame such as the one returned by `NERLoader.get_links_for_review`.
        Classifier is also saved as `NERLoader.clf`.

        Args:
            train_data (pd.DataFrame): [description]
            ent_mention_col (str, optional): [description]. Defaults to "ent_text".
            ent_type_col (str, optional): [description]. Defaults to "ent_label".
            ent_context_col (str, optional): [description]. Defaults to "item_description".
            candidate_title_col (str, optional): [description]. Defaults to "candidate_title".
            candidate_type_col (str, optional): [description]. Defaults to "candidate_type".
            candidate_context_col (str, optional): [description]. Defaults to "candidate_description".
            target_col (str, optional): [description]. Defaults to "link_correct".
            linking_classifier (BaseEstimator, optional): scikit-learn classifier. Must have a `fit(X, y, **kwargs)` method. Defaults to MLPClassifier.
            classifier_kwargs (dict, optional): kwargs to pass into `linking_classifier`. If a "random_state" value is not set, it's set as the value of the `random_seed` argument.
                Defaults to {"random_state": 42, "max_iter": 1000}.
            random_seed (int, optional): Used in all random generators. Defaults to 42.

        Returns:
            BaseEstimator: trained classifier
        """

        logger.info("Training entity linker...")

        extra_kwargs = {}
        if sbert_model is not None:
            extra_kwargs.update({"sbert_model": sbert_model})
        if suffix_list is not None:
            extra_kwargs.update({"suffix_list": suffix_list})

        if "random_state" not in classifier_kwargs.keys():
            classifier_kwargs["random_state"] = random_seed

        nel_pipeline = Pipeline(
            [
                ("featgen", NELFeatureGenerator()),
                ("classifier", linking_classifier(**classifier_kwargs)),
            ]
        )

        y_true = get_target_values_from_review_data(train_data, target_col)

        nel_pipeline = nel_pipeline.fit(
            train_data,
            y=y_true,
            featgen__ent_mention_col=ent_mention_col,
            featgen__ent_type_col=ent_type_col,
            featgen__ent_context_col=ent_context_col,
            featgen__candidate_title_col=candidate_title_col,
            featgen__candidate_type_col=candidate_type_col,
            featgen__candidate_context_col=candidate_context_col,
        )

        self.clf = nel_pipeline

        return self.clf

    def _get_ner_model(self, model_type):
        """Get best spacy NER model"""
        return best_spacy_pipeline.load_model(model_type)

    def get_list_of_entities_from_es(
        self,
        model_type: str,
        limit: int = None,
        random_sample: bool = True,
        random_seed: int = 42,
        spacy_batch_size: int = 128,
        spacy_no_processes: int = 1,
        ignore_duplicated_ents: bool = True,
    ) -> List[dict]:
        """
        Run NER on entities in the source (Heritage Connector) index and add the results back to the index using the
        HC namespace.

        Args:
            model_type (str): spaCy model type
            limit (int, optional): limit number of documents to process. Ideal for testing. Defaults to None.
            random_sample (bool, optional): Whether to randomly sample documents from the Elasticsearch index.
                Only has an effect if limit is not None. Defaults to True.
            random_seed (int, optional): random seed for `random_sample`. Defaults to 42.
            spacy_batch_size (int, optional): batch size for spaCy's `nlp.pipe`. Defaults to 128.
            spacy_no_processes (int, optional): n_process for spaCy's `nlp.pipe`. Defaults to 1.

        Returns:
            List[dict]: The current state of `entity_list`.
        """

        logger.info(f"Fetching docs and running NER.")

        doc_list = list(
            self._get_doc_generator(
                self.source_index, limit, random_sample, random_seed
            )
        )
        self.nlp = self._get_ner_model(model_type)

        if ignore_duplicated_ents and (
            not spacy.tokens.Span.has_extension("entity_duplicate")
        ):
            logger.warn(
                "Parameter `ignore_duplicate_ents` has been set to True for `NERLoader.add_ner_entities_to_es()` but spaCy spans have no `entity_duplicate` attribute. "
                "You can resolve this by adding the `duplicate_entity_detector` component from hc_nlp.pipeline to the end of your spaCy pipeline. "
                "For now, the detection of duplicate entity mentions in a document will be disabled."
            )
            ignore_duplicated_ents = False

        # list of {"item_uri": _, "ent_label": _, "ent_text": _} triples
        entity_list = []
        uris = [item[0] for item in doc_list]
        descriptions = [item[1] for item in doc_list]

        spacy_docs = list(
            tqdm(
                self.nlp.pipe(
                    descriptions,
                    batch_size=spacy_batch_size,
                    n_process=spacy_no_processes,
                )
            )
        )

        for idx, doc in enumerate(spacy_docs):
            entity_list += self._spacy_doc_to_ent_list(
                uris[idx], descriptions[idx], doc, ignore_duplicated_ents
            )

        self._entity_list = entity_list

        return self.entity_list

    def load_entities_into_es_no_links(self):
        logger.info(
            f"Loading {len(self._entity_list)} entities into {self.source_index}"
        )

        """create a DataFrame from a list of (uri, entity label, entity text) triples, and load it into the ES index one entity label at a time"""
        entity_triples_df = pd.DataFrame(self._entity_list)

        for entity_label in entity_triples_df["ent_label"].unique():
            # logger.debug(f"label {entity_label}..")
            # this is the same as HC.entityLABEL e.g. HC.entityPERSON
            rdf_predicate = HC["entity" + entity_label]
            ent_label_df = entity_triples_df[
                entity_triples_df["ent_label"] == entity_label
            ]
            self.record_loader.add_triples(
                ent_label_df,
                predicate=rdf_predicate,
                subject_col="item_uri",
                object_col="ent_text",
                object_is_uri=False,
                progress_bar=False,
            )

    def load_entities_into_es(self, linking_confidence_threshold: float = 0.5):
        logger.info(
            f"Loading {len(self._entity_list)} entities into {self.source_index}"
        )

        if "link_candidates" not in pd.DataFrame(self._entity_list).columns:
            self.load_entities_into_es_no_links()
            return

        entity_df = self.entity_list_as_dataframe
        entities_with_link_candidates, entities_without_link_candidates = (
            entity_df[~entity_df["candidate_rank"].isna()],
            entity_df[entity_df["candidate_rank"].isna()],
        )
        entity_triples_unlinked = []
        entity_triples_linked = []

        # TODO: remove training data and add in ground truth values

        # predict True links for entities with link candidates
        logger.info(f"Predicting links for entity annotations with link candidates")
        link_y_pred = self.clf.predict_proba(entities_with_link_candidates)[:, 1]
        entities_with_link_candidates["y_pred_proba"] = link_y_pred
        entities_with_link_candidates["y_pred"] = (
            entities_with_link_candidates["y_pred_proba"]
            >= linking_confidence_threshold
        )

        for _, group in entities_with_link_candidates.groupby(
            ["item_uri", "item_description_with_ent"]
        ):
            group_uri = group["item_uri"].iloc[0]
            group_ent_type = group["ent_label"].iloc[0]
            group_ent_text = group["ent_text"].iloc[0]
            group_rdf_predicate = HC["entity" + group_ent_type]

            if sum(group["y_pred"]) > 0:
                # there is a record that has been predicted to be a link for this entity mention
                group_pos_candidates = group.loc[
                    group["y_pred"] == True, "candidate_uri"  # noqa: E712
                ]

                entity_triples_linked += [
                    {
                        "item_uri": group_uri,
                        "rdf_predicate": group_rdf_predicate,
                        "linked_value": uri,
                    }
                    for uri in group_pos_candidates
                ]

            else:
                # there is no predicted linked record, so just add the text
                entity_triples_unlinked.append(
                    {
                        "item_uri": group_uri,
                        "rdf_predicate": group_rdf_predicate,
                        "entity_text": group_ent_text,
                    }
                )

        # add text entities in for entities with no link candidates
        for _, row in entities_without_link_candidates.iterrows():
            entity_triples_unlinked.append(
                {
                    "item_uri": row["item_uri"],
                    "rdf_predicate": HC["entity" + row["ent_label"]],
                    "entity_text": row["ent_text"],
                }
            )

        # load into JSON-LD index
        logger.info(
            f"Loading {len(entity_triples_linked)} linked and {len(entity_triples_unlinked)} unlinked entities into {self.source_index}"
        )
        for _, group in pd.DataFrame(entity_triples_linked).groupby("rdf_predicate"):
            self.record_loader.add_triples(
                group,
                predicate=group["rdf_predicate"].iloc[0],
                subject_col="item_uri",
                object_col="linked_value",
                object_is_uri=True,
                progress_bar=False,
            )

        for _, group in pd.DataFrame(entity_triples_unlinked).groupby("rdf_predicate"):
            self.record_loader.add_triples(
                group,
                predicate=group["rdf_predicate"].iloc[0],
                subject_col="item_uri",
                object_col="entity_text",
                object_is_uri=False,
                progress_bar=False,
            )

    def get_link_candidates(self, candidates_per_entity_mention: int) -> List[dict]:
        """Get link candidates for each of the items in `entity_list` by searching the entity mention in
        the target Elasticsearch index. Only searches for link candidates for entities with types specified in `entity_types_to_link`,
        and excludes any candidates with a URI which is the same as the URI of the source entity.

        Args:
            entity_list (List[dict]): each item has the form `{"item_uri": _, "ent_label": _, "ent_text": _,}`

        Returns:
            List[dict]: The current state of `entity_list`. Each item has the form `{"item_uri": _, "ent_label": _, "ent_text": _,}`
                (not in types to link) or `{"item_uri": _, "ent_label": _, "ent_text": _, "link_candidates": [{"uri": _, "label": _,
                "description": _}, ...]}` (in types to link).
        """

        if not self._entity_list:
            raise ValueError(
                "Entities have not yet been retrieved from the Elasticsearch index. Run `get_list_of_entities_from_es` first."
            )

        entity_list_with_link_candidates = []
        logger.info(
            f"Getting link candidates for each of {len(self._entity_list)} entities"
        )
        for item in tqdm(self._entity_list):
            if item["ent_label"] in self.entity_types_to_link:
                link_candidates = self._search_es_for_entity_mention(
                    item["ent_text"],
                    n=candidates_per_entity_mention * 2,
                    reduce_to_key_fields=True,
                )
                link_candidates = [
                    i for i in link_candidates if i["uri"] != item["item_uri"]
                ][:candidates_per_entity_mention]
                entity_list_with_link_candidates.append(
                    dict(item, **{"link_candidates": link_candidates})
                )
            else:
                entity_list_with_link_candidates.append(item)

        self._entity_list = entity_list_with_link_candidates

        return self._entity_list

    def _search_es_for_entity_mention(
        self, mention: str, n: int, reduce_to_key_fields: bool = True
    ) -> List[dict]:
        """
        Given an entity mention, search the target Elasticsearch fields and return up to `n` documents.
        """

        # empty string for a field in multi_match query doesn't register as a field
        search_fields = [
            self.target_fields["title"],
            self.target_fields.get("alias", ""),
        ]

        # this query boosts exact matches (type: phrase) over fuzzy matches, which don't return exact matches first
        # because of Elasticsearch field analysis.
        query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": mention,
                                "type": "phrase",
                                "fields": search_fields,
                                "boost": 10,
                            }
                        },
                        {
                            "multi_match": {
                                "query": mention,
                                "type": "best_fields",
                                "fields": search_fields,
                                "fuzziness": "AUTO",
                            }
                        },
                    ]
                }
            }
        }

        search_results = (
            es.search(
                index=self.target_index,
                body=query,
                size=n,
            )
            .get("hits", {})
            .get("hits", [])
        )

        if reduce_to_key_fields:
            return [self._reduce_doc_to_key_fields(i) for i in search_results]

        return search_results

    def _get_dict_field_from_dot_notation(
        self, doc: dict, field_dot_notation: str
    ) -> dict:
        """Get a field from a dictonary from Elasticsearch dot notation."""

        nested_field = doc["_source"]
        fields_split = field_dot_notation.split(".")
        if fields_split[0] != "graph" and fields_split[-1] == "@value":
            fields_split = fields_split[0:-1]

        if field_dot_notation.startswith("data") and "." in field_dot_notation[5:]:
            fields_split = ["data", field_dot_notation[5:]]

        for idx, field in enumerate(fields_split):
            if idx + 1 < len(fields_split):
                nested_field = nested_field.get(field, {})
            else:
                nested_field = nested_field.get(field, "")

        return nested_field

    def _reduce_doc_to_key_fields(self, doc: dict) -> dict:
        """Reduce doc to target_uri, target_title_field, target_description_field, target_alias_field"""

        # key_fields = set(["uri"] + list(self.target_fields.values()))

        reduced_doc = {"uri": self._get_dict_field_from_dot_notation(doc, "uri")}

        for target_field_name, target_field in self.target_fields.items():
            if target_field_name == "description":
                target_field_value = self.text_preprocess_func(
                    self._get_dict_field_from_dot_notation(doc, target_field)
                )
            else:
                target_field_value = self._get_dict_field_from_dot_notation(
                    doc, target_field
                )

            reduced_doc.update({target_field_name: target_field_value})

        return {k: v for k, v in reduced_doc.items() if v != ""}

    def _spacy_doc_to_ent_list(
        self,
        item_uri: str,
        item_description: str,
        doc: spacy.tokens.Doc,
        ignore_duplicated_ents: bool,
    ) -> List[dict]:
        """
        Convert a spaCy doc with entities found into a list of dictionaries.
        """

        ent_data_list = []

        if ignore_duplicated_ents:
            ent_is_suitable = lambda ent: (ent.label_ in self.entity_types) and (
                ent._.entity_duplicate is False
            )
        else:
            ent_is_suitable = lambda ent: ent.label_ in self.entity_types

        for ent in doc.ents:
            if ent_is_suitable(ent):
                ent_data_list.append(
                    {
                        "item_uri": item_uri,
                        "item_description": item_description,
                        "item_description_with_ent": item_description[
                            : doc[ent.start].idx
                        ]
                        + self.entity_markers[0]
                        + ent.text
                        + self.entity_markers[1]
                        + item_description[doc[ent.start].idx + len(ent.text) :],
                        "ent_label": ent.label_,
                        "ent_text": ent.text,
                        "ent_sentence": ent.sent.text,
                        "ent_start_idx": doc[ent.start].idx,
                        "ent_end_idx": doc[ent.start].idx + len(ent.text),
                    }
                )

        return ent_data_list

    def _get_doc_generator(
        self,
        index: str,
        limit: Optional[int] = None,
        random_sample: bool = True,
        random_seed: int = 42,
    ) -> Generator[List[Tuple[str, str]], None, None]:
        """
        Returns a generator of document IDs and descriptions from the Elasticsearch index, batched according to
            `self.batch_size` and limited according to `limit`. Only documents with an XSD.description value are
            returned.

        Args:
            limit (Optional[int], optional): limit the number of documents to get and therefore load. Defaults to None.
            random_sample (bool, optional): whether to take documents at random. Defaults to True.
            random_seed (int, optional): random seed to use if random sampling is enabled using the `random_sample` parameter. Defaults to 42.

        Returns:
            Generator[List[Tuple[str, str]]]: generator of lists with length `self.batch_size`, where each list contains `(uri, description)` tuples.
        """

        # TODO: make description field follow source description from class instance and rename function.

        if random_sample:
            es_query = {
                "query": {
                    "function_score": {
                        "query": {
                            "bool": {
                                "must": [
                                    {
                                        "exists": {
                                            "field": "data.http://www.w3.org/2001/XMLSchema#description"
                                        }
                                    },
                                ]
                            }
                        },
                        "random_score": {"seed": random_seed, "field": "_seq_no"},
                    }
                }
            }
        else:
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "exists": {
                                    "field": "data.http://www.w3.org/2001/XMLSchema#description"
                                }
                            },
                        ]
                    }
                }
            }

        doc_generator = helpers.scan(
            client=es,
            index=index,
            query=es_query,
            preserve_order=True,
        )

        if limit:
            doc_generator = islice(doc_generator, limit)

        doc_generator = (
            (
                doc["_id"],
                self.text_preprocess_func(
                    doc["_source"]["data"][
                        "http://www.w3.org/2001/XMLSchema#description"
                    ]
                ),
            )
            for doc in doc_generator
        )

        return doc_generator
