from heritageconnector.disambiguation import compare_fields


# TODO: write test for compare_fields.compare


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
