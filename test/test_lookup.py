#  Tests for methods in heritageconnector.lookup

from heritageconnector.entity_matching import lookup


class TestUrlMethods:
    """
    Test individual methods to extract Wikidata IDs from URLs.
    """

    def test_from_wikipedia(self):
        qcode = lookup.wikidata_id.from_wikipedia(
            "https://en.wikipedia.org/wiki/Joseph_Lister"
        )
        assert qcode == "Q155768"

    def test_from_getty(self):
        qcode = lookup.wikidata_id.from_getty(
            "https://www.getty.edu/vow/ULANFullDisplay?find=Wheldon&role=&nation=&prev_page=1&subjectid=500044753"
        )

        qcode2 = lookup.wikidata_id.from_getty(
            "http://vocab.getty.edu/page/ulan/500044753"
        )

        assert qcode == qcode2 == "Q21498264"

    def test_from_britannica(self):
        qcode = lookup.wikidata_id.from_britannica(
            "https://www.britannica.com/biography/Joseph-Marie-Jacquard"
        )

        assert qcode == "Q310833"


class TestGet:
    """
    Test wikidata_id.get_from_url() method with different URLS - ie that URL routing is working.
    """

    def test_get_oxdnb(self):
        qcode = lookup.wikidata_id.get_from_url(
            "https://www.oxforddnb.com/view/article/23105"
        )
        assert qcode == "Q1338141"

    def test_get_wikipedia(self):
        qcode = lookup.wikidata_id.get_from_url("en.wikipedia.org/wiki/Joseph_Lister")
        assert qcode == "Q155768"

    def test_get_graces_guide(self):
        qcode = lookup.wikidata_id.get_from_url(
            "https://www.gracesguide.co.uk/Benn_Brothers_(Publishers)"
        )
        assert qcode == "Q5392786"

    def test_get_national_archives(self):
        qcode = lookup.wikidata_id.get_from_url(
            "http://discovery.nationalarchives.gov.uk/details/c/F41403"
        )
        assert qcode == "Q725786"

    def test_get_viaf(self):
        qcode = lookup.wikidata_id.get_from_url("https://viaf.org/viaf/40191108/")
        assert qcode == "Q2539929"

    def test_get_multiple(self):
        get_wid = lookup.wikidata_id()
        qcodes, urls = get_wid.get_from_free_text(
            """
            ODNB: J. F. Payne, ‘Lettsom, John Coakley (1744–1815)’, rev. Roy Porter, Oxford Dictionary of National Biography, Oxford University Press, 2004 
            [http://www.oxforddnb.com/view/article/16527, accessed 6 Jan 2015] John Coakley Lettsom (1744–1815): doi:10.1093/ref:odnb/16527
            WIKI: http://en.wikipedia.org/wiki/John_C._Lettsome
            """,
            return_urls=True,
        )

        assert qcodes == ["Q6224512"]
        assert urls == [
            "http://www.oxforddnb.com/view/article/16527",
            "http://en.wikipedia.org/wiki/John_C._Lettsome",
        ]


class TestCustomExpressions:
    def test_regex_ulan(self):
        get_wid = lookup.wikidata_id(
            custom_patterns=[
                (
                    r"(?:Union List of Artist Names Online|ULAN)[A-Za-z\s:.]+ID:\s?(500\d{6})",
                    "P245",
                ),
            ]
        )
        qcodes, _ = get_wid.get_from_free_text(
            """
            ULAN: ID: 500011683; 
            Maurice William Greiffenhagen (1862–1931): doi:10.1093/ref:odnb/33558
            Sir William Beechey (1753–1839): doi:10.1093/ref:odnb/1949 Anne Phyllis Beechey, Lady Beechey (1764–1833): doi:10.1093/ref:odnb/75533; Union List of Artist Names Online: The J. Paul Getty Trust: ID: 500014785

            Sir William Beechey (1753–1839): doi:10.1093/ref:odnb/1949.  Anne Phyllis Beechey, Lady Beechey (1764–1833): doi:10.1093/ref:odnb/75533

            """
        )
        assert qcodes == ["Q6793159", "Q48566"]
