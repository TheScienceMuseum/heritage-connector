#  Tests for methods in heritageconnector.lookup

from heritageconnector.lookup import from_url


class TestLookup:
    def test_from_oxdnb(self):
        qcode = from_url.wikidata_id.from_oxdnb(
            "https://www.oxforddnb.com/view/article/23105"
        )
        assert qcode == "Q1338141"

    def test_from_wikipedia(self):
        qcode = from_url.wikidata_id.from_wikipedia(
            "https://en.wikipedia.org/wiki/Joseph_Lister"
        )
        assert qcode == "Q155768"
