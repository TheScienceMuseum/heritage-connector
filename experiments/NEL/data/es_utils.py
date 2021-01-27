from elasticsearch import Elasticsearch, helpers
from itertools import islice


class ElasticsearchConnector:
    def __init__(self, es, index):
        self.es = es
        self.es_index = index

    def get_document_generator(self, limit: int = None, batch_size: int = 50):
        """
        Returns paginated generator where each document is represented by a (uri, description) tuple.
        Each page is a list of `batch_size` tuples.
        """

        query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {"exists": {"field": "graph.@rdfs:label"}},
                                {
                                    "exists": {
                                        "field": "data.http://www.w3.org/2001/XMLSchema#description"
                                    }
                                },
                            ]
                        }
                    },
                    "random_score": {"seed": 42, "field": "_seq_no"},
                }
            }
        }

        doc_generator = helpers.scan(
            client=self.es,
            index=self.es_index,
            query=query,
            preserve_order=True,
        )

        if limit:
            doc_generator = islice(doc_generator, limit)

        doc_generator = (simplify_document(doc) for doc in doc_generator)
        doc_generator = ((doc["uri"], doc["description"]) for doc in doc_generator)

        doc_generator = paginate_generator(doc_generator, batch_size)

        return doc_generator


def simplify_document(doc: dict, max_description_length: int = None) -> dict:
    """
    Extracts just the URI, topconcept, label and description from an Elasticsearch document.
    """

    description = doc["_source"]["data"].get(
        "http://www.w3.org/2001/XMLSchema#description", ""
    )

    if max_description_length is not None:
        description = description[0:max_description_length]

    return {
        "uri": doc["_id"],
        "topconcept": doc["_source"]["graph"]["@skos:hasTopConcept"]["@value"],
        "label": doc["_source"]["graph"].get("@rdfs:label", {}).get("@value", ""),
        "description": description,
    }


def paginate_generator(generator, page_size: int):
    """
    Returns an iterator that returns items from the provided generator grouped into `page_size`.
    If the size of the output from the original generator isn't an exact multiple of
    `page_size`, the last list returned by the iterator will be of size less than `page_size`.

    Returns:
        iterator of lists
    """
    return iter(lambda: list(islice(generator, page_size)), [])


def get_hc_candidates(es, text, limit=5):
    body = {
        "query": {
            "match": {"graph.@rdfs:label.@value": {"query": text, "fuzziness": "AUTO"}}
        }
    }

    res = es.search(index="heritageconnector", body=body, size=limit,)[
        "hits"
    ]["hits"]

    return [simplify_document(item, max_description_length=140) for item in res]


def get_wiki_candidates(es, text, limit=5):
    body = {
        "query": {"match": {"labels_aliases": {"query": text, "fuzziness": "AUTO"}}}
    }

    res = es.search(index="wikidump2", body=body, size=limit,)[
        "hits"
    ]["hits"]

    return [item["_source"] for item in res]
