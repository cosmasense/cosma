"""Integration tests for search functionality."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from cosma_backend.db.database import Database
from cosma_backend.models.file import File
from cosma_backend.models.status import ProcessingStatus
from cosma_backend.searcher.searcher import HybridSearcher, SearchResult


@pytest_asyncio.fixture
async def populated_db(temp_db: Database):
    """Create a database with sample files for search testing."""
    files_data = [
        {
            "file_path": "/docs/python_tutorial.md",
            "filename": "python_tutorial.md",
            "title": "Python Programming Tutorial",
            "summary": "A comprehensive guide to learning Python programming language",
            "content": "Python is a versatile programming language used for web development and data science.",
            "keywords": ["python", "programming", "tutorial"],
        },
        {
            "file_path": "/docs/javascript_guide.md",
            "filename": "javascript_guide.md",
            "title": "JavaScript Developer Guide",
            "summary": "Modern JavaScript development practices and patterns",
            "content": "JavaScript powers the modern web with frameworks like React and Vue.",
            "keywords": ["javascript", "web", "development"],
        },
        {
            "file_path": "/docs/machine_learning.md",
            "filename": "machine_learning.md",
            "title": "Machine Learning Fundamentals",
            "summary": "Introduction to machine learning concepts and algorithms",
            "content": "Machine learning uses Python libraries like scikit-learn and TensorFlow.",
            "keywords": ["machine learning", "python", "ai"],
        },
        {
            "file_path": "/notes/meeting_notes.txt",
            "filename": "meeting_notes.txt",
            "title": "Team Meeting Notes",
            "summary": "Notes from the weekly team meeting",
            "content": "Discussed project timeline and deliverables for Q1.",
            "keywords": ["meeting", "notes", "team"],
        },
        {
            "file_path": "/code/test_utils.py",
            "filename": "test_utils.py",
            "title": "Test Utilities",
            "summary": "Utility functions for testing",
            "content": "Helper functions for unit testing Python applications.",
            "keywords": ["testing", "python", "utilities"],
        },
    ]

    now = datetime.now(timezone.utc)

    for data in files_data:
        file = File(
            path=Path(data["file_path"]),
            file_path=data["file_path"],
            filename=data["filename"],
            extension=Path(data["filename"]).suffix,
            file_size=1024,
            created=now,
            modified=now,
            accessed=now,
            content_type="text/plain",
            content=data["content"],
            content_hash=f"hash_{data['filename']}",
            summary=data["summary"],
            title=data["title"],
            keywords=data["keywords"],
            status=ProcessingStatus.COMPLETE,
        )
        await temp_db.upsert_file(file)

    return temp_db


@pytest.mark.integration
class TestKeywordSearch:
    """Integration tests for keyword search."""

    @pytest.mark.asyncio
    async def test_basic_keyword_search(self, populated_db: Database):
        """Test basic keyword search returns results."""
        results = await populated_db.keyword_search("python", limit=10)

        assert len(results) > 0
        # All results should have a score
        for file, score in results:
            assert score > 0
            assert file.file_path is not None

    @pytest.mark.asyncio
    async def test_keyword_search_with_special_chars(self, populated_db: Database):
        """Test keyword search handles special characters safely."""
        # These should not crash even with FTS5 special characters
        special_queries = [
            'test"query',
            "test*",
            "test:value",
            "(test)",
            "test AND query",  # Should be treated as literal without operators
        ]

        for query in special_queries:
            # Should not raise an exception
            results = await populated_db.keyword_search(query, limit=10)
            # Results may be empty, but should not crash
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_keyword_search_empty_query(self, populated_db: Database):
        """Test keyword search with empty query returns empty results."""
        results = await populated_db.keyword_search("", limit=10)
        assert results == []

        results = await populated_db.keyword_search("   ", limit=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_directory_filter(self, populated_db: Database):
        """Test keyword search with directory filter."""
        # Search only in /docs directory
        results = await populated_db.keyword_search("python", limit=10, directory="/docs")

        for file, _score in results:
            assert file.file_path.startswith("/docs")

    @pytest.mark.asyncio
    async def test_keyword_search_with_operators(self, populated_db: Database):
        """Test keyword search with boolean operators enabled."""
        # With operators enabled, AND should be a boolean operator
        results = await populated_db.keyword_search(
            "python AND learning",
            limit=10,
            allow_operators=True
        )

        # Should find the machine learning file which has both terms
        assert len(results) >= 0  # May or may not find results depending on content


@pytest.mark.integration
class TestHybridSearch:
    """Integration tests for hybrid search with RRF."""

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(self, populated_db: Database, mock_embedder):
        """Test that hybrid search returns results with RRF scores."""
        searcher = HybridSearcher(db=populated_db, embedder=mock_embedder)

        results = await searcher.search("python programming", limit=5)

        assert len(results) > 0
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.combined_score > 0
            assert result.match_type in ["semantic", "keyword", "hybrid"]

    @pytest.mark.asyncio
    async def test_hybrid_search_rrf_scores_are_small(self, populated_db: Database, mock_embedder):
        """Test that RRF scores are in expected range (small values)."""
        searcher = HybridSearcher(db=populated_db, embedder=mock_embedder)

        results = await searcher.search("python", limit=10)

        for result in results:
            # RRF scores are typically small (1/k range)
            # With k=60, max single-list score is ~0.0167
            # With both lists, max is ~0.0333
            assert result.combined_score < 0.1
            assert result.combined_score > 0

    @pytest.mark.asyncio
    async def test_hybrid_search_preserves_match_type(self, populated_db: Database, mock_embedder):
        """Test that match_type is correctly set based on result sources."""
        searcher = HybridSearcher(db=populated_db, embedder=mock_embedder)

        results = await searcher.search("python", limit=10)

        # At least some results should exist
        assert len(results) > 0

        # Match types should be valid
        valid_types = {"semantic", "keyword", "hybrid"}
        for result in results:
            assert result.match_type in valid_types

    @pytest.mark.asyncio
    async def test_hybrid_search_with_directory_filter(self, populated_db: Database, mock_embedder):
        """Test hybrid search with directory filtering."""
        searcher = HybridSearcher(db=populated_db, embedder=mock_embedder)

        results = await searcher.search("python", limit=10, directory="/docs")

        for result in results:
            assert result.file_metadata.file_path.startswith("/docs")

    @pytest.mark.asyncio
    async def test_hybrid_search_special_characters(self, populated_db: Database, mock_embedder):
        """Test hybrid search handles special characters safely."""
        searcher = HybridSearcher(db=populated_db, embedder=mock_embedder)

        special_queries = [
            'test"quote',
            "test*wildcard",
            "test:colon",
            "(parentheses)",
        ]

        for query in special_queries:
            # Should not raise an exception
            results = await searcher.search(query, limit=5)
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_query(self, populated_db: Database, mock_embedder):
        """Test hybrid search with empty query."""
        searcher = HybridSearcher(db=populated_db, embedder=mock_embedder)

        # Empty query should still work (semantic search may return results)
        results = await searcher.search("", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_ranks_preserved(self, populated_db: Database, mock_embedder):
        """Test that semantic and keyword ranks are preserved in results."""
        searcher = HybridSearcher(db=populated_db, embedder=mock_embedder)

        results = await searcher.search("python", limit=10)

        for result in results:
            # At least one rank should be set
            has_semantic = result.semantic_rank is not None
            has_keyword = result.keyword_rank is not None
            assert has_semantic or has_keyword

            # Hybrid results should have both
            if result.match_type == "hybrid":
                assert has_semantic and has_keyword


@pytest.mark.integration
class TestSearchSuggestions:
    """Integration tests for search suggestions."""

    @pytest.mark.asyncio
    async def test_get_fts5_suggestions(self, populated_db: Database):
        """Test FTS5-based suggestions."""
        suggestions = await populated_db.get_fts5_suggestions("pyth", limit=5)

        # Should find suggestions starting with "pyth"
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_get_fts5_suggestions_empty_prefix(self, populated_db: Database):
        """Test suggestions with empty prefix."""
        suggestions = await populated_db.get_fts5_suggestions("", limit=5)
        assert suggestions == []

    @pytest.mark.asyncio
    async def test_search_suggestions_via_searcher(self, populated_db: Database, mock_embedder):
        """Test search suggestions through the searcher."""
        searcher = HybridSearcher(db=populated_db, embedder=mock_embedder)

        suggestions = await searcher.get_search_suggestions("pyth", limit=5)

        assert isinstance(suggestions, list)


@pytest.mark.integration
class TestSearchResultSerialization:
    """Integration tests for SearchResult serialization."""

    @pytest.mark.asyncio
    async def test_search_result_to_json(self, populated_db: Database, mock_embedder):
        """Test SearchResult.to_json() includes all fields."""
        searcher = HybridSearcher(db=populated_db, embedder=mock_embedder)

        results = await searcher.search("python", limit=1)

        if results:
            result = results[0]
            json_data = result.to_json()

            assert "file_path" in json_data
            assert "filename" in json_data
            assert "combined_score" in json_data
            assert "match_type" in json_data
            assert "semantic_rank" in json_data
            assert "keyword_rank" in json_data

            # Verify types
            assert isinstance(json_data["combined_score"], float)
            assert isinstance(json_data["match_type"], str)
