from elasticsearch import helpers
from elasticsearch import Elasticsearch
import rdflib
from rdflib import Graph, Literal, URIRef
from rdflib.serializer import Serializer
import json
import os
from typing import Generator, List, Tuple, Optional
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
from heritageconnector.utils.generic import paginate_generator
from heritageconnector.config import config
from heritageconnector import logging, errors, best_spacy_pipeline
import pandas as pd
import spacy

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
        """

        generator = self._record_update_generator(
            records, predicate, subject_col, object_col, object_is_uri
        )
        es_bulk(generator, len(records))

    def _record_update_generator(
        self,
        df: pd.DataFrame,
        predicate: rdflib.URIRef,
        subject_col: str = "SUBJECT",
        object_col: str = "OBJECT",
        object_is_uri: bool = True,
    ) -> Generator[dict, None, None]:
        """Yields JSON-LD docs for the `add_triples` method, to update existing records with new triples"""

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


def es_bulk(action_generator, total_iterations=None):
    """Batch load a set of new records into ElasticSearch"""

    successes = 0
    errs = []

    for ok, action in tqdm(
        helpers.parallel_bulk(
            client=es,
            index=index,
            actions=action_generator,
            chunk_size=es_config["chunk_size"],
            queue_size=es_config["queue_size"],
            raise_on_error=False,
        ),
        total=total_iterations,
    ):
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


def delete(id):
    """Delete an existing ElasticSearch record"""

    es.delete(id)


def get_by_uri(uri):
    """Return an existing ElasticSearch record"""

    res = es.search(index=index, body={"query": {"term": {"uri.keyword": uri}}})
    if len(res["hits"]["hits"]):
        return res["hits"]["hits"][0]


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
        batch_size: Optional[int] = 1024,
        entity_types: List[str] = [
            "PERSON",
            "ORG",
            "NORP",
            "FAC",
            "LOC",
            "OBJECT",
            "LANGUAGE",
            "DATE",
        ],
    ):
        """
        Initialise instance of NERLoader.

        Args:
            record_loader (RecordLoader): instance of RecordLoader, with parameters suitable for the current Heritage Connector index.
            batch_size (Optional[int], optional): size of batches to process documents in. Defaults to 1024.
            entity_types (List[str], optional): entity types extracted from the spaCy model.
        """

        self.record_loader = record_loader
        self.es_index = index
        self.batch_size = batch_size
        self.entity_types = entity_types

    def _get_ner_model(self, model_type):
        """Get best spacy NER model"""
        return best_spacy_pipeline.load_model(model_type)

    def fetch_es_docs_and_run_ner(
        self,
        model_type: str,
        limit: int = None,
        random_sample: bool = True,
        random_seed: int = 42,
        spacy_batch_size: int = 128,
        spacy_no_processes: int = 1,
    ) -> List[dict]:

        logger.info(
            f"fetching docs and running NER on them in batches of {self.batch_size}"
        )

        doc_generator = self._get_doc_generator(limit, random_sample, random_seed)
        self.nlp = self._get_ner_model(model_type)

        # create list of {"item_uri": _, "ent_label": _, "ent_text": _} triples for loading into ES
        triples_list = []

        for batch in tqdm(doc_generator, unit="batch", total=limit):
            descriptions = [item[1] for item in batch]
            spacy_doc_batch = list(
                self.nlp.pipe(
                    descriptions,
                    batch_size=spacy_batch_size,
                    n_process=spacy_no_processes,
                )
            )

            for idx, doc in enumerate(spacy_doc_batch):
                triples_list += self._spacy_doc_to_dataframe(batch[idx][0], doc)

        return triples_list

    def load_ner_results_into_es(self, triples_list: List[dict]):
        logger.info(
            f"loading {len(triples_list)} entities into the {index} index by label"
        )
        # create a DataFrame from this list of triples, and load it into the ES index one entity label at a time
        entity_triples_df = pd.DataFrame(triples_list)

        for entity_label in entity_triples_df["ent_label"].unique():
            logger.debug(f"label {entity_label}..")
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
            )

    def _spacy_doc_to_dataframe(
        self, item_uri: str, doc: spacy.tokens.Doc
    ) -> pd.DataFrame:
        """
        Convert batch of spaCy docs with generated entities into a DataFrame on which `record_loader.add_triples()` can
        be called.
        """

        ent_data_list = []

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                ent_data_list.append(
                    {
                        "item_uri": item_uri,
                        "ent_label": ent.label_,
                        "ent_text": ent.text,
                    }
                )

        # return pd.DataFrame(ent_data_list)
        return ent_data_list

    def _get_doc_generator(
        self,
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
            index=self.es_index,
            query=es_query,
            preserve_order=True,
        )

        if limit:
            doc_generator = islice(doc_generator, limit)

        doc_generator = (
            (
                doc["_id"],
                doc["_source"]["data"]["http://www.w3.org/2001/XMLSchema#description"],
            )
            for doc in doc_generator
        )
        doc_generator = paginate_generator(doc_generator, self.batch_size)

        return doc_generator

    def add_ner_entities_to_es(self, model_type: str, entity_types: List[str]):
        """
        Add entities from an NER model to the Heritage Connector Elasticsearch index.

        Args:
            model_type (str): [description]
            entity_types (List[str]): [description]
        """
        pass
