import sys

sys.path.append("../../..")

from heritageconnector import datastore
from typing import Optional, Callable, List, Iterable, Tuple
from elasticsearch import helpers


class ESEntityLinker:
    def __init__(
        self,
        es,
        es_index: str,
        nlp,
        source_description_field: str,
        target_title_field: str,
        target_description_field: str,
        target_alias_field: str = None,
    ):
        """
        Each of the input variables is dot notation for accessing the field from Elasticsearch `_source`,
        e.g. 'graph.@rdfs:label.@value'.
        """

        self.es = es
        self.es_index = es_index

        self.nlp = nlp

        self.source_fields = {"description": source_description_field}

        self.target_fields = {
            "title": target_title_field,
            "description": target_description_field,
        }

        if target_alias_field is not None:
            self.target_fields.update(
                {
                    "alias": target_alias_field,
                }
            )

    def get_link_candidates_from_spacy_doc(
        self, spacy_doc, n_per_ent: int
    ) -> List[dict]:
        """
        Return details of candidates for each entity mention in a spaCy doc.
        """

        link_candidates = {}

        for ent in spacy_doc.ents:
            ent_results = self._search_es_for_mention(
                ent.text, n_per_ent, reduce_to_key_fields=True
            )
            link_candidates[ent.text] = ent_results

        return link_candidates

    def _search_es_for_mention(
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

        query = {
            "query": {
                "multi_match": {
                    "query": mention,
                    "fuzziness": "AUTO",
                    "fields": search_fields,
                }
            }
        }

        search_results = (
            self.es.search(
                index=self.es_index,
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
        """Reduce doc to uri, source_description_field, target_title_field, target_description_field, target_alias_field"""

        key_fields = set(
            ["uri"]
            + list(self.source_fields.values())
            + list(self.target_fields.values())
        )

        reduced_doc = {
            field: self._get_dict_field_from_dot_notation(doc, field)
            for field in key_fields
        }

        return {k: v for k, v in reduced_doc.items() if v != ""}
