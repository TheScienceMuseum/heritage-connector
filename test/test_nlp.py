from heritageconnector.nlp.string_pairs import fuzzy_match


def test_string_pairs():
    assert fuzzy_match("abc", "abc") is True
    assert fuzzy_match("brian", "brain", threshold=0.5) is True
    assert fuzzy_match("abc", "xyz") is False
