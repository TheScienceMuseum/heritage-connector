from heritageconnector.disambiguation import compare_fields
from heritageconnector.namespace import WD
from rdflib import URIRef, Literal
import pytest


def test_compare():
    with pytest.raises(ValueError):
        # first argument must either be URIRef or Literal
        assert compare_fields.compare("Arthur", "Q670277", "Arthur Russell") == 1

    # comparing entities
    assert compare_fields.compare(WD.Q670277, "Q670277", "Arthur Russell") == 1

    # comparing Literal to label (string)
    assert (
        compare_fields.compare(Literal("Arthur Russell"), "Q670277", "Arthur Russell")
        == 1
    )

    # comparing Literal to entity value (float)
    assert compare_fields.compare(Literal(4), "4", "label") == 1


class TestSimilarities:
    def test_string_similarity(self):
        assert compare_fields.similarity_string("abc", "abc") == 1

        assert compare_fields.similarity_string("abc", "xyz") == 0

        assert compare_fields.similarity_string(["brian", "rohit"], "rohit") == 1

        assert compare_fields.similarity_string(["a", "b", "c"], ["d", "e", "f"]) == 0

    def test_numeric_similarity(self):
        assert compare_fields.similarity_numeric(100, 50) == 50 / 100

    def test_categorical_similarity(self):
        assert (
            compare_fields.similarity_categorical(
                ["apple", "banana"], ["apple", "orange"]
            )
            == 1
        )

        assert compare_fields.similarity_categorical("apple", "orange") == 0

        assert (
            compare_fields.similarity_categorical(
                ["apple", "orange"], "orange", raise_on_diff_types=False
            )
            == 1
        )
