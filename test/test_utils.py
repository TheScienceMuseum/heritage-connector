from heritageconnector.utils import wikidata, generic
import time
import os


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
