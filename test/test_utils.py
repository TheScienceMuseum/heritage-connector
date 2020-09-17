from heritageconnector.utils import wikidata
import time


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
        _ = next(wikidata.get_distance_between_entities_cached({"Q5", "Q10"}))
        mid = time.time()
        _ = next(wikidata.get_distance_between_entities_cached({"Q5", "Q10"}))
        end = time.time()

        assert (end - mid) < (mid - start)
        assert (end - mid) < 2
