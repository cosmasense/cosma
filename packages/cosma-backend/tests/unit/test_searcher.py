"""Unit tests for search functionality including FTS5 query parsing and RRF."""

import pytest

from cosma_backend.searcher.fts5_query import (
    escape_fts5_token,
    parse_fts5_query,
    build_prefix_query,
)
from cosma_backend.searcher.rrf import compute_rrf_scores, merge_with_rrf


@pytest.mark.unit
class TestFTS5QueryParser:
    """Test cases for FTS5 query parsing and escaping."""

    def test_escape_empty_token(self):
        """Test escaping an empty token."""
        assert escape_fts5_token("") == ""
        assert escape_fts5_token(None) == ""

    def test_escape_simple_token(self):
        """Test escaping a simple token without special characters."""
        assert escape_fts5_token("hello") == '"hello"'
        assert escape_fts5_token("world") == '"world"'

    def test_escape_token_with_quotes(self):
        """Test escaping a token containing double quotes."""
        assert escape_fts5_token('test"quote') == '"test""quote"'
        assert escape_fts5_token('"quoted"') == '"""quoted"""'

    def test_escape_token_with_special_chars(self):
        """Test escaping tokens with FTS5 special characters."""
        # These are wrapped in quotes which makes them literal
        assert escape_fts5_token("test*") == '"test*"'
        assert escape_fts5_token("test:value") == '"test:value"'
        assert escape_fts5_token("(parens)") == '"(parens)"'

    def test_parse_empty_query(self):
        """Test parsing an empty query."""
        assert parse_fts5_query("") == ""
        assert parse_fts5_query("   ") == ""
        assert parse_fts5_query(None) == ""

    def test_parse_single_term(self):
        """Test parsing a single search term."""
        assert parse_fts5_query("hello") == '"hello"'
        assert parse_fts5_query("  hello  ") == '"hello"'

    def test_parse_multiple_terms(self):
        """Test parsing multiple search terms."""
        assert parse_fts5_query("hello world") == '"hello" "world"'
        assert parse_fts5_query("one two three") == '"one" "two" "three"'

    def test_parse_with_special_characters(self):
        """Test parsing queries with special characters."""
        # Special chars should be escaped by wrapping in quotes
        assert parse_fts5_query('test"query') == '"test""query"'
        assert parse_fts5_query("test*") == '"test*"'
        assert parse_fts5_query("test:value") == '"test:value"'

    def test_parse_with_operators_disabled(self):
        """Test that operators are escaped when allow_operators=False."""
        # AND, OR, NOT should be treated as literal terms
        assert parse_fts5_query("test AND query") == '"test" "AND" "query"'
        assert parse_fts5_query("foo OR bar") == '"foo" "OR" "bar"'

    def test_parse_with_operators_enabled(self):
        """Test that operators are preserved when allow_operators=True."""
        assert parse_fts5_query("test AND query", allow_operators=True) == '"test" AND "query"'
        assert parse_fts5_query("foo OR bar", allow_operators=True) == '"foo" OR "bar"'
        assert parse_fts5_query("NOT excluded", allow_operators=True) == 'NOT "excluded"'

    def test_parse_mixed_operators_and_terms(self):
        """Test parsing mixed operators and terms."""
        result = parse_fts5_query("hello AND world OR test", allow_operators=True)
        assert result == '"hello" AND "world" OR "test"'

    def test_build_prefix_query_empty(self):
        """Test building prefix query with empty input."""
        assert build_prefix_query("") == ""
        assert build_prefix_query("   ") == ""

    def test_build_prefix_query_simple(self):
        """Test building a simple prefix query."""
        assert build_prefix_query("test") == '"test"*'
        assert build_prefix_query("hello") == '"hello"*'

    def test_build_prefix_query_strips_special_chars(self):
        """Test that prefix query strips special characters."""
        assert build_prefix_query("test*") == '"test"*'
        assert build_prefix_query('test"') == '"test"*'
        assert build_prefix_query("test:") == '"test"*'


@pytest.mark.unit
class TestRRF:
    """Test cases for Reciprocal Rank Fusion."""

    def test_compute_rrf_empty_lists(self):
        """Test RRF with empty lists."""
        scores = compute_rrf_scores([])
        assert scores == {}

        scores = compute_rrf_scores([[]])
        assert scores == {}

    def test_compute_rrf_single_list(self):
        """Test RRF with a single ranked list."""
        ranked_list = [("a", "data_a"), ("b", "data_b"), ("c", "data_c")]
        scores = compute_rrf_scores([ranked_list], k=60)

        # First item: 1/(60+0) = 1/60
        # Second item: 1/(60+1) = 1/61
        # Third item: 1/(60+2) = 1/62
        assert abs(scores["a"] - 1/60) < 0.0001
        assert abs(scores["b"] - 1/61) < 0.0001
        assert abs(scores["c"] - 1/62) < 0.0001

    def test_compute_rrf_multiple_lists(self):
        """Test RRF with multiple ranked lists."""
        list1 = [("a", "data"), ("b", "data"), ("c", "data")]
        list2 = [("b", "data"), ("a", "data"), ("d", "data")]

        scores = compute_rrf_scores([list1, list2], k=60)

        # "a": 1/60 + 1/61 (rank 0 in list1, rank 1 in list2)
        # "b": 1/61 + 1/60 (rank 1 in list1, rank 0 in list2)
        # "c": 1/62 (only in list1 at rank 2)
        # "d": 1/62 (only in list2 at rank 2)

        expected_a = 1/60 + 1/61
        expected_b = 1/61 + 1/60
        expected_c = 1/62
        expected_d = 1/62

        assert abs(scores["a"] - expected_a) < 0.0001
        assert abs(scores["b"] - expected_b) < 0.0001
        assert abs(scores["c"] - expected_c) < 0.0001
        assert abs(scores["d"] - expected_d) < 0.0001

        # a and b should have the same score (symmetric)
        assert abs(scores["a"] - scores["b"]) < 0.0001

    def test_compute_rrf_items_in_both_lists_rank_higher(self):
        """Test that items appearing in both lists get higher RRF scores."""
        list1 = [("shared", "data"), ("only_in_1", "data")]
        list2 = [("shared", "data"), ("only_in_2", "data")]

        scores = compute_rrf_scores([list1, list2], k=60)

        # shared appears in both lists at rank 0: 1/60 + 1/60 = 2/60
        # others appear only in one list: 1/61 each
        assert scores["shared"] > scores["only_in_1"]
        assert scores["shared"] > scores["only_in_2"]

    def test_merge_with_rrf_empty_inputs(self):
        """Test merge_with_rrf with empty inputs."""
        result = merge_with_rrf([], [])
        assert result == []

    def test_merge_with_rrf_semantic_only(self):
        """Test merge_with_rrf with only semantic results."""
        semantic = [(1, "file_1", 0.5), (2, "file_2", 0.7)]
        keyword = []

        result = merge_with_rrf(semantic, keyword, k=60)

        assert len(result) == 2
        # Check first result
        assert result[0][0] == 1  # id
        assert result[0][1] == "file_1"  # data
        assert result[0][3] == "semantic"  # match_type
        assert result[0][4] == 0  # semantic_rank
        assert result[0][5] is None  # keyword_rank

    def test_merge_with_rrf_keyword_only(self):
        """Test merge_with_rrf with only keyword results."""
        semantic = []
        keyword = [(1, "file_1", 0.9), (2, "file_2", 0.8)]

        result = merge_with_rrf(semantic, keyword, k=60)

        assert len(result) == 2
        assert result[0][3] == "keyword"  # match_type
        assert result[0][4] is None  # semantic_rank
        assert result[0][5] == 0  # keyword_rank

    def test_merge_with_rrf_hybrid_results(self):
        """Test merge_with_rrf with overlapping results."""
        semantic = [(1, "file_1", 0.3), (2, "file_2", 0.5), (3, "file_3", 0.7)]
        keyword = [(2, "file_2", 0.9), (4, "file_4", 0.8), (1, "file_1", 0.7)]

        result = merge_with_rrf(semantic, keyword, k=60)

        # Create lookup by id for easier assertions
        result_by_id = {r[0]: r for r in result}

        # file_1 and file_2 appear in both - should be hybrid
        assert result_by_id[1][3] == "hybrid"
        assert result_by_id[2][3] == "hybrid"

        # file_3 only in semantic
        assert result_by_id[3][3] == "semantic"

        # file_4 only in keyword
        assert result_by_id[4][3] == "keyword"

        # Check ranks are preserved
        assert result_by_id[1][4] == 0  # semantic_rank for file_1
        assert result_by_id[1][5] == 2  # keyword_rank for file_1
        assert result_by_id[2][4] == 1  # semantic_rank for file_2
        assert result_by_id[2][5] == 0  # keyword_rank for file_2

    def test_merge_with_rrf_sorting(self):
        """Test that merge_with_rrf returns results sorted by RRF score."""
        # file_2 appears first in keyword, second in semantic
        # file_1 appears first in semantic, third in keyword
        # file_2 should rank higher due to better average position
        semantic = [(1, "file_1", 0.3), (2, "file_2", 0.5)]
        keyword = [(2, "file_2", 0.9), (3, "file_3", 0.8), (1, "file_1", 0.7)]

        result = merge_with_rrf(semantic, keyword, k=60)

        # file_2: 1/61 + 1/60 = top scorer (rank 1 in semantic, rank 0 in keyword)
        # file_1: 1/60 + 1/62 = second (rank 0 in semantic, rank 2 in keyword)
        # file_3: 1/61 = third (only in keyword at rank 1)

        assert result[0][0] == 2  # file_2 should be first
        assert result[1][0] == 1  # file_1 should be second
        assert result[2][0] == 3  # file_3 should be third

    def test_merge_with_rrf_k_parameter(self):
        """Test that different k values affect scores but not relative ranking."""
        semantic = [(1, "file_1", 0.3), (2, "file_2", 0.5)]
        keyword = [(2, "file_2", 0.9), (1, "file_1", 0.8)]

        result_k60 = merge_with_rrf(semantic, keyword, k=60)
        result_k10 = merge_with_rrf(semantic, keyword, k=10)

        # Relative ordering should be the same
        assert [r[0] for r in result_k60] == [r[0] for r in result_k10]

        # But scores should be different (higher with lower k)
        assert result_k10[0][2] > result_k60[0][2]
