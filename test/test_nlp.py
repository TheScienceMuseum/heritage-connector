from heritageconnector.nlp.string_pairs import fuzzy_match, fuzzy_match_lists


def test_string_pairs():
    assert fuzzy_match("abc", "abc") is True
    assert fuzzy_match("brian", "brain", threshold=0.5) is True
    assert fuzzy_match("abc", "xyz") is False

    assert fuzzy_match_lists("abc", "xyz") is False
    assert fuzzy_match_lists(["abc", ""], "xyz") is False
    assert fuzzy_match_lists(["abc", "brian"], ["xyz", "brain"], threshold=0.5) is True
