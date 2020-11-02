from heritageconnector.utils import wikidata, generic
import time
import os
import pytest


class TestWikidataUtils:
    def test_get_distance_between_entities(self):
        # distance between the same entity should be zero
        assert wikidata.get_distance_between_entities({"Q5", "Q5"}) == 0

        # these entities are greater than 10 links apart, so reciprocal=True should return zero
        assert (
            wikidata.get_distance_between_entities(
                {"Q5", "Q100"}, reciprocal=True, max_path_length=10
            )
            < 0.01
        )

    def test_get_distance_between_entities_cached(self):
        """Same as above but cached"""

        start = time.time()
        _ = wikidata.get_distance_between_entities_cached({"Q5", "Q10"})
        mid = time.time()
        _ = wikidata.get_distance_between_entities_cached({"Q5", "Q10"})
        end = time.time()

        assert (end - mid) < (mid - start)
        assert (end - mid) < 0.4

    def test_filter_qids_in_class_tree(self):
        filtered_qids = wikidata.filter_qids_in_class_tree(
            ["Q83463949", "Q83463824", "Q16917", "Q15944511"], "Q43229"
        )

        assert isinstance(filtered_qids, list)
        assert len(filtered_qids) == 2

    def test_filter_qids_in_class_tree_multiple(self):
        filtered_qids_2 = wikidata.filter_qids_in_class_tree(
            ["Q83463949", "Q83463824", "Q16917", "Q15944511", "Q12823105"],
            ["Q43229", "Q618123"],
        )

        assert isinstance(filtered_qids_2, list)
        assert len(filtered_qids_2) == 3

    def test_get_wikidata_equivalents_for_properties(self):
        properties = [
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://xmlns.com/foaf/0.1/maker",
            "http://no_wikidata_equivalent.org",
            "http://www.wikidata.org/prop/direct/P21",
        ]

        # running with raise_missing=True should raise an error as one URI doesn't have a Wikidata equivalent
        with pytest.raises(ValueError):
            _ = wikidata.get_wikidata_equivalents_for_properties(
                properties, raise_missing=True
            )

        results = wikidata.get_wikidata_equivalents_for_properties(
            properties, raise_missing=False
        )

        assert len(results.keys()) == len(properties)
        assert all(["wikidata" in v for v in results.values() if v is not None])
        assert results["http://no_wikidata_equivalent.org"] is None
        assert (
            results["http://www.wikidata.org/prop/direct/P21"]
            == "http://www.wikidata.org/prop/direct/P21"
        )


class TestGenericUtils:
    def test_cache(self):
        @generic.cache("./cache")
        def multiply_slowly_cached(a, b):
            time.sleep(2)
            return a * b

        start = time.time()
        multiply_slowly_cached(5, 10)
        mid = time.time()
        multiply_slowly_cached(5, 10)
        end = time.time()

        os.remove("cache")

        assert end - mid < mid - start

    def test_paginate_list(self):
        original_list = [1, 2, 3, 4, 5, 6, 7]
        new_list = generic.paginate_list(original_list, page_size=3)

        assert new_list == [[1, 2, 3], [4, 5, 6], [7]]
