"""Pin the glob→LIKE conversion so search filter behavior stays
predictable. The converter is tiny but used in 4 different DB
methods, so a single source of truth is worth a small test."""

import pytest

from cosma_backend.utils.glob_sql import glob_to_like, is_glob


@pytest.mark.unit
class TestGlobToLike:
    def test_star_becomes_percent(self):
        assert glob_to_like("*.pdf") == "%.pdf"
        assert glob_to_like("*report*") == "%report%"

    def test_question_becomes_underscore(self):
        assert glob_to_like("file?.txt") == "file_.txt"

    def test_literal_percent_is_escaped(self):
        # User typing "50%" should not turn into a SQL wildcard.
        assert glob_to_like("50%") == "50!%"

    def test_literal_underscore_is_escaped(self):
        assert glob_to_like("snake_case") == "snake!_case"

    def test_literal_escape_char_is_escaped(self):
        # The escape char itself ('!') must be doubled so a literal
        # "wow!" doesn't accidentally escape the next character.
        assert glob_to_like("wow!") == "wow!!"

    def test_empty_passes_through(self):
        assert glob_to_like("") == ""


@pytest.mark.unit
class TestIsGlob:
    def test_detects_star(self):
        assert is_glob("*.pdf") is True

    def test_detects_question(self):
        assert is_glob("file?.txt") is True

    def test_plain_string_is_not_a_glob(self):
        assert is_glob("report.pdf") is False
