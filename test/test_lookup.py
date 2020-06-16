#  Tests for methods in heritageconnector.lookup

from heritageconnector.lookup import from_url


class TestUrlMethods:
    """
    Test individual methods to extract Wikidata IDs from URLs.
    """

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

    def test_from_getty(self):
        qcode = from_url.wikidata_id.from_getty(
            "https://www.getty.edu/vow/ULANFullDisplay?find=Wheldon&role=&nation=&prev_page=1&subjectid=500044753"
        )

        qcode2 = from_url.wikidata_id.from_getty(
            "http://vocab.getty.edu/page/ulan/500044753"
        )

        assert qcode == qcode2 == "Q21498264"

    def test_from_gracesguide(self):
        qcode = from_url.wikidata_id.from_graces_guide(
            "https://www.gracesguide.co.uk/Benn_Brothers_(Publishers)"
        )

        assert qcode == "Q5392786"


class TestGet:
    """
    Test wikidata_id.get() method with different URLS - ie that URL routing is working.
    """

    def test_get_oxdnb(self):
        qcode = from_url.wikidata_id.get("https://www.oxforddnb.com/view/article/23105")
        assert qcode == "Q1338141"

    def test_get_wikipedia(self):
        qcode = from_url.wikidata_id.get("en.wikipedia.org/wiki/Joseph_Lister")
        assert qcode == "Q155768"
