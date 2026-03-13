#!/usr/bin/env python
"""
@File    :   searcher.py
@Time    :   2025/07/14
@Author  :
@Version :   1.0
@Contact :
@License :
@Desc    :   Hybrid search combining semantic similarity and keyword matching
"""

from dataclasses import dataclass


from cosma_backend.db.database import Database
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.logging import get_logger
from cosma_backend.models import File
from cosma_backend.searcher.rrf import merge_with_rrf

# Configure logger
logger = get_logger(__name__)


class SearchError(Exception):
    """Base exception for search errors."""


@dataclass
class SearchResult:
    """
    Represents a search result with metadata and scoring.

    Scores are computed using Reciprocal Rank Fusion (RRF) which combines
    rankings from multiple search methods without requiring score normalization.
    """
    file_metadata: File
    combined_score: float = 0.0             # RRF score (set by searcher)
    match_type: str = "unknown"             # Type of match: semantic, keyword, hybrid
    semantic_rank: int | None = None        # Rank in semantic results (for debugging)
    keyword_rank: int | None = None         # Rank in keyword results (for debugging)
    
    def to_json(self) -> dict:
        """
        Convert SearchResult to JSON-serializable dictionary.

        Returns:
            Dictionary representation of the search result
        """
        return {
            "file_path": str(self.file_metadata.path),
            "filename": str(self.file_metadata.filename),
            "combined_score": self.combined_score,
            "match_type": self.match_type,
            "semantic_rank": self.semantic_rank,
            "keyword_rank": self.keyword_rank,
        }


class HybridSearcher:
    """
    Hybrid search engine combining semantic similarity and keyword matching.
    """

    def __init__(self, db: Database, embedder: AutoEmbedder | None = None) -> None:
        """
        Initialize hybrid searcher.

        Args:
            db: Database instance
            embedder: Embedder for generating query embeddings
        """
        self.db = db
        self.embedder = embedder or AutoEmbedder()
        logger.info("HybridSearcher initialized")

    async def search(
        self,
        query: str,
        limit: int = 20,
        semantic_weight: float = 0.7,  # Deprecated, kept for backward compatibility
        keyword_weight: float = 0.3,   # Deprecated, kept for backward compatibility
        semantic_threshold: float = 2.0,
        include_metadata: bool = True,
        directory: str | None = None,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching using RRF.

        Uses Reciprocal Rank Fusion (RRF) to combine results from semantic and
        keyword search. RRF is rank-based and doesn't require score normalization.

        Args:
            query: Search query
            limit: Maximum number of results
            semantic_weight: Deprecated, ignored (kept for backward compatibility)
            keyword_weight: Deprecated, ignored (kept for backward compatibility)
            semantic_threshold: Maximum distance for semantic matches
            include_metadata: Include file metadata in results
            directory: Optional directory path to limit search scope
            rrf_k: RRF smoothing constant (default 60)

        Returns:
            List of SearchResult objects sorted by RRF score
        """
        logger.info("Performing hybrid search with RRF",
                       query=query,
                       limit=limit,
                       rrf_k=rrf_k,
                       directory=directory)

        # Fetch more results than needed - RRF benefits from larger candidate pools
        fetch_limit = limit * 3

        # Collect results from both search methods
        semantic_list: list[tuple[int, File, float]] = []
        keyword_list: list[tuple[int, File, float]] = []

        # 1. Semantic search
        try:
            semantic_matches = await self._semantic_search(query, fetch_limit, semantic_threshold, directory)
            for file_metadata, distance in semantic_matches:
                file_id = file_metadata.id if hasattr(file_metadata, "id") else hash(file_metadata.file_path)
                semantic_list.append((file_id, file_metadata, distance))

            logger.debug("Semantic search completed", results=len(semantic_list))
        except Exception as e:
            logger.warning("Semantic search failed", error=str(e))

        # 2. Keyword search
        try:
            keyword_matches = await self._keyword_search(query, fetch_limit, directory)
            for file_metadata, score in keyword_matches:
                file_id = file_metadata.id if hasattr(file_metadata, "id") else hash(file_metadata.file_path)
                keyword_list.append((file_id, file_metadata, score))

            logger.debug("Keyword search completed", results=len(keyword_list))
        except Exception as e:
            logger.warning("Keyword search failed", error=str(e))

        # 3. Merge using RRF
        merged = merge_with_rrf(semantic_list, keyword_list, k=rrf_k)

        # 4. Build SearchResult objects
        results = []
        for file_id, file_metadata, rrf_score, match_type, sem_rank, kw_rank in merged[:limit]:
            result = SearchResult(
                file_metadata=file_metadata,
                combined_score=rrf_score,
                match_type=match_type,
                semantic_rank=sem_rank,
                keyword_rank=kw_rank,
            )
            results.append(result)

        logger.info("Hybrid search completed",
                       total_results=len(results),
                       semantic_matches=len(semantic_list),
                       keyword_matches=len(keyword_list))

        return results

    async def _semantic_search(self, query: str, limit: int, threshold: float, directory: str | None = None) -> list[tuple]:
        """Perform semantic search using embeddings."""
        if not self.embedder:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)

            # Search similar files
            return await self.db.search_similar_files(
                query_embedding=query_embedding,
                limit=limit,
                threshold=threshold,
                directory=directory
            )


        except Exception as e:
            logger.exception("Semantic search failed", error=str(e))
            return []

    async def _keyword_search(self, query: str, limit: int, directory: str | None = None) -> list[tuple]:
        """Perform keyword search using SQLite FTS5."""
        try:
            # Use database FTS5 for efficient keyword search
            results = await self.db.keyword_search(query, limit * 2, directory)
            
            # Results are already in (file_metadata, score) format
            # FTS5 returns relevance scores that work well for ranking
            return results

        except Exception as e:
            logger.exception("Keyword search failed", error=str(e))
            return []

    async def search_similar_to_file(self,
                                   file_id: int,
                                   limit: int = 10,
                                   threshold: float = 0.8) -> list[SearchResult]:
        """
        Find files similar to a given file using its embedding.

        Args:
            file_id: ID of the reference file
            limit: Maximum number of results
            threshold: Similarity threshold

        Returns:
            List of similar files
        """
        logger.info("Searching for similar files", file_id=file_id)

        try:
            # Get file's embedding
            embedding_data = await self.db.get_file_embedding(file_id)
            if not embedding_data:
                msg = f"No embedding found for file {file_id}"
                raise SearchError(msg)

            embedding, model_name, dimensions = embedding_data

            # Search for similar files
            results = await self.db.search_similar_files(
                query_embedding=embedding,
                limit=limit + 1,  # +1 to account for the file itself
                threshold=threshold
            )

            # Convert to SearchResult objects, excluding the original file
            search_results = []
            for rank, (file_metadata, distance) in enumerate(results):
                if hasattr(file_metadata, "id") and file_metadata.id != file_id:
                    # For similar-to-file search, use simple rank-based scoring
                    result = SearchResult(
                        file_metadata=file_metadata,
                        combined_score=1.0 / (60 + rank),  # RRF-style score
                        match_type="semantic",
                        semantic_rank=rank,
                    )
                    search_results.append(result)

            # Limit results
            search_results = search_results[:limit]

            logger.info("Found similar files",
                           file_id=file_id,
                           similar_count=len(search_results))

            return search_results

        except Exception as e:
            logger.exception("Similar file search failed",
                                file_id=file_id,
                                error=str(e))
            msg = f"Failed to find similar files: {e!s}"
            raise SearchError(msg)

    async def get_search_suggestions(self, query: str, limit: int = 5) -> list[str]:
        """
        Get search suggestions using FTS5 prefix matching.

        Args:
            query: Partial query (prefix)
            limit: Maximum number of suggestions

        Returns:
            List of suggested search terms
        """
        logger.debug("Getting search suggestions", query=query)

        if not query or not query.strip():
            return []

        try:
            # Use FTS5 prefix search for efficient autocomplete
            suggestions = await self.db.get_fts5_suggestions(query.strip(), limit)

            logger.debug("Generated search suggestions",
                            query=query,
                            suggestions=len(suggestions))

            return suggestions

        except Exception as e:
            logger.exception("Failed to get search suggestions",
                                query=query,
                                error=str(e))
            return []


# Convenience function for simple search
async def search_files(db: Database,
                      query: str,
                      limit: int = 20,
                      search_type: str = "hybrid") -> list[SearchResult]:
    """
    Convenience function for file search.

    Args:
        db: Database instance
        query: Search query
        limit: Maximum results
        search_type: "hybrid", "semantic", or "keyword"

    Returns:
        List of search results
    """
    searcher = HybridSearcher(db)

    if search_type == "hybrid":
        return await searcher.search(query, limit, semantic_threshold=1.5)
    if search_type == "semantic":
        return await searcher.search(query, limit, semantic_weight=1.0, keyword_weight=0.0, semantic_threshold=1.5)
    if search_type == "keyword":
        return await searcher.search(query, limit, semantic_weight=0.0, keyword_weight=1.0, semantic_threshold=1.5)
    msg = f"Invalid search_type: {search_type}"
    raise ValueError(msg)
