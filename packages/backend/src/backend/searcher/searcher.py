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

import logging

from backend.db.database import Database
from backend.embedder.embedder import AutoEmbedder
from backend.logging import sm
from backend.models import File

# Configure logger
logger = logging.getLogger(__name__)


class SearchError(Exception):
    """Base exception for search errors."""


@dataclass
class SearchResult:
    """
    Represents a search result with metadata and scoring.
    """
    file_metadata: File
    semantic_score: float | None = None  # Distance from semantic search (lower is better)
    keyword_score: float | None = None   # Keyword match score (higher is better)
    combined_score: float = 0.0             # Combined weighted score
    match_type: str = "unknown"             # Type of match: semantic, keyword, hybrid

    def __post_init__(self):
        """Calculate combined score after initialization."""
        if self.semantic_score is not None and self.keyword_score is not None:
            self.match_type = "hybrid"
            # Normalize semantic score (convert distance to similarity)
            semantic_similarity = max(0, 1 - self.semantic_score)
            # Combine with weights (adjust as needed)
            self.combined_score = (0.7 * semantic_similarity) + (0.3 * self.keyword_score)
        elif self.semantic_score is not None:
            self.match_type = "semantic"
            self.combined_score = max(0, 1 - self.semantic_score)
        elif self.keyword_score is not None:
            self.match_type = "keyword"
            self.combined_score = self.keyword_score
        else:
            self.combined_score = 0.0
    
    def to_json(self) -> dict:
        """
        Convert SearchResult to JSON-serializable dictionary.

        Returns:
            Dictionary representation of the search result
        """
        return {
            "file_path": str(self.file_metadata.path),
            "filename": str(self.file_metadata.filename),
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "combined_score": self.combined_score,
            "match_type": self.match_type
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
        logger.info(sm("HybridSearcher initialized"))

    async def search(self,
                    query: str,
                    limit: int = 20,
                    semantic_weight: float = 0.7,
                    keyword_weight: float = 0.3,
                    semantic_threshold: float = 2.0,
                    include_metadata: bool = True,
                    directory: str | None = None) -> list[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Args:
            query: Search query
            limit: Maximum number of results
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            semantic_threshold: Maximum distance for semantic matches
            include_metadata: Include file metadata in results
            directory: Optional directory path to limit search scope

        Returns:
            List of SearchResult objects sorted by combined score
        """
        logger.info(sm("Performing hybrid search",
                       query=query,
                       limit=limit,
                       semantic_weight=semantic_weight,
                       keyword_weight=keyword_weight,
                       directory=directory))

        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        if total_weight > 0:
            semantic_weight /= total_weight
            keyword_weight /= total_weight

        # Collect results from both search methods
        semantic_results = {}
        keyword_results = {}

        # 1. Semantic search
        try:
            semantic_matches = await self._semantic_search(query, limit * 2, semantic_threshold, directory)
            for file_metadata, distance in semantic_matches:
                file_id = file_metadata.id if hasattr(file_metadata, "id") else hash(file_metadata.file_path)
                semantic_results[file_id] = (file_metadata, distance)

            logger.debug(sm("Semantic search completed", results=len(semantic_results)))
        except Exception as e:
            logger.warning(sm("Semantic search failed", error=str(e)))

        # 2. Keyword search
        try:
            keyword_matches = await self._keyword_search(query, limit * 2, directory)
            for file_metadata, score in keyword_matches:
                file_id = file_metadata.id if hasattr(file_metadata, "id") else hash(file_metadata.file_path)
                keyword_results[file_id] = (file_metadata, score)

            logger.debug(sm("Keyword search completed", results=len(keyword_results)))
        except Exception as e:
            logger.warning(sm("Keyword search failed", error=str(e)))

        # 3. Combine results
        combined_results = []
        all_file_ids = set(semantic_results.keys()) | set(keyword_results.keys())

        for file_id in all_file_ids:
            semantic_data = semantic_results.get(file_id)
            keyword_data = keyword_results.get(file_id)

            # Get file metadata (prefer from semantic search for completeness)
            if semantic_data:
                file_metadata = semantic_data[0]
                semantic_score = semantic_data[1]
            else:
                file_metadata = keyword_data[0]
                semantic_score = None

            keyword_score = keyword_data[1] if keyword_data else None

            # Create search result
            result = SearchResult(
                file_metadata=file_metadata,
                semantic_score=semantic_score,
                keyword_score=keyword_score
            )

            # Apply improved additive scoring algorithm
            semantic_component = 0.0
            keyword_component = 0.0
            
            # Semantic component: Convert distance to similarity and normalize
            if result.semantic_score is not None:
                # Convert distance (lower=better) to similarity (higher=better)
                # Use exponential decay to emphasize closer matches
                import math
                semantic_similarity = math.exp(-result.semantic_score)  # Range: 0-1, closer=higher
                semantic_component = semantic_similarity * 0.5  # Scale to 0-0.5 range
                
            # Keyword component: Direct score
            if result.keyword_score is not None:
                keyword_component = result.keyword_score * 0.5  # Scale to 0-0.5 range
                
            # Combined score: Sum of components (not weighted average)
            result.combined_score = semantic_component + keyword_component
            
            # Determine match type based on components
            if semantic_component > 0 and keyword_component > 0:
                result.match_type = "hybrid"
            elif keyword_component > 0:
                result.match_type = "keyword"  
            elif semantic_component > 0:
                result.match_type = "semantic"
            else:
                result.match_type = "none"

            combined_results.append(result)

        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)

        # Limit results
        final_results = combined_results[:limit]

        logger.info(sm("Hybrid search completed",
                       total_results=len(final_results),
                       semantic_matches=len(semantic_results),
                       keyword_matches=len(keyword_results)))

        return final_results

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
            logger.exception(sm("Semantic search failed", error=str(e)))
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
            logger.exception(sm("Keyword search failed", error=str(e)))
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
        logger.info(sm("Searching for similar files", file_id=file_id))

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
            for file_metadata, distance in results:
                if hasattr(file_metadata, "id") and file_metadata.id != file_id:
                    result = SearchResult(
                        file_metadata=file_metadata,
                        semantic_score=distance,
                        match_type="semantic"
                    )
                    search_results.append(result)

            # Limit results
            search_results = search_results[:limit]

            logger.info(sm("Found similar files",
                           file_id=file_id,
                           similar_count=len(search_results)))

            return search_results

        except Exception as e:
            logger.exception(sm("Similar file search failed",
                                file_id=file_id,
                                error=str(e)))
            msg = f"Failed to find similar files: {e!s}"
            raise SearchError(msg)

    async def get_search_suggestions(self, query: str, limit: int = 5) -> list[str]:
        """
        Get search suggestions based on existing keywords and file titles.

        Args:
            query: Partial query
            limit: Maximum number of suggestions

        Returns:
            List of suggested search terms
        """
        logger.debug(sm("Getting search suggestions", query=query))

        try:
            # Get all files to extract keywords and titles
            files = await self.db.get_files(limit=1000)  # Reasonable limit

            suggestions = set()
            query_lower = query.lower()

            for file_metadata in files:
                # Add matching keywords
                if file_metadata.keywords:
                    for keyword in file_metadata.keywords:
                        if keyword.lower().startswith(query_lower):
                            suggestions.add(keyword)

                # Add matching title words
                if file_metadata.title:
                    for word in file_metadata.title.split():
                        if word.lower().startswith(query_lower) and len(word) > 2:
                            suggestions.add(word)

                # Stop if we have enough suggestions
                if len(suggestions) >= limit * 2:
                    break

            # Sort suggestions by relevance (simple alphabetical for now)
            sorted_suggestions = sorted(suggestions)[:limit]

            logger.debug(sm("Generated search suggestions",
                            query=query,
                            suggestions=len(sorted_suggestions)))

            return sorted_suggestions

        except Exception as e:
            logger.exception(sm("Failed to get search suggestions",
                                query=query,
                                error=str(e)))
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
