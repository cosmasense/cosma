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

import asyncio
import time
from dataclasses import dataclass
from typing import Literal


from cosma_backend.db.database import Database
from cosma_backend.embedder import AutoEmbedder
from cosma_backend.logging import get_logger
from cosma_backend.models import Application, File
from cosma_backend.searcher.rrf import compute_rrf_scores, merge_with_rrf


SearchScope = Literal["all", "files", "applications"]

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


@dataclass
class ApplicationResult:
    """A single application search hit.

    Apps don't get RRF-fused with files — they live in their own
    visual group in the UI (a divider between the two lists), so a
    raw BM25-relevance score is enough. We don't have semantic
    embeddings for apps yet; if the LLM enrichment in Step 4 adds
    them, this struct can grow a `semantic_rank` like SearchResult.
    """
    application: Application
    relevance_score: float = 0.0


@dataclass
class SearchBundle:
    """Result aggregate from a hybrid search.

    Separate file and application lists so the frontend can render
    the two groups distinctly without re-classifying or re-sorting.
    `apps` may be empty even when files matched (and vice versa);
    callers should not assume one implies the other.
    """
    files: list["SearchResult"]
    apps: list[ApplicationResult]
    total_ms: float = 0.0


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
        path_pattern: str | None = None,
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
        # Timing chain — every stage logs its own ms so a slow first
        # search has a single grep-friendly trail. Pre-fix, we had no
        # idea whether a 30s timeout was the embed call, the FTS5 query,
        # the rerank, or the response serialization.
        t_start = time.perf_counter()
        logger.info("Search request received",
                       query=query,
                       limit=limit,
                       rrf_k=rrf_k,
                       directory=directory)

        # Fetch more results than needed - RRF benefits from larger candidate pools
        fetch_limit = limit * 3

        # Collect results from both search methods
        semantic_list: list[tuple[int, File, float]] = []
        keyword_list: list[tuple[int, File, float]] = []

        # Semantic + Keyword run in parallel — independent work, no
        # reason to serialize on the event loop. The legacy search()
        # returns files only, so the apps task that used to run here
        # has moved to search_with_apps where it participates in the
        # unified cross-modal RRF.
        t_lookup = time.perf_counter()
        sem_task = asyncio.create_task(self._semantic_search(query, fetch_limit, semantic_threshold, directory, path_pattern))
        kw_task = asyncio.create_task(self._keyword_search(query, fetch_limit, directory, path_pattern))
        sem_result, kw_result = await asyncio.gather(
            sem_task, kw_task, return_exceptions=True,
        )
        lookup_ms = (time.perf_counter() - t_lookup) * 1000.0

        if isinstance(sem_result, Exception):
            logger.warning("Semantic search failed", error=str(sem_result))
        else:
            for file_metadata, distance in sem_result:
                file_id = file_metadata.id if hasattr(file_metadata, "id") else hash(file_metadata.file_path)
                semantic_list.append((file_id, file_metadata, distance))

        if isinstance(kw_result, Exception):
            logger.warning("Keyword search failed", error=str(kw_result))
        else:
            for file_metadata, score in kw_result:
                file_id = file_metadata.id if hasattr(file_metadata, "id") else hash(file_metadata.file_path)
                keyword_list.append((file_id, file_metadata, score))

        # 3. Merge using RRF
        t_rrf = time.perf_counter()
        merged = merge_with_rrf(semantic_list, keyword_list, k=rrf_k)
        rrf_ms = (time.perf_counter() - t_rrf) * 1000.0

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

        total_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info("Search completed",
                       query=query,
                       total_ms=round(total_ms, 1),
                       lookup_ms=round(lookup_ms, 1),
                       rrf_ms=round(rrf_ms, 1),
                       results=len(results),
                       semantic_matches=len(semantic_list),
                       keyword_matches=len(keyword_list))

        return results

    async def _semantic_search(self, query: str, limit: int, threshold: float, directory: str | None = None, path_pattern: str | None = None) -> list[tuple]:
        """Perform semantic search using embeddings."""
        if not self.embedder:
            return []

        try:
            # Generate query embedding off the event loop. The previous
            # synchronous `embed_text(query)` call blocked asyncio while the
            # SentenceTransformer ran a 50-200 ms forward pass on CPU — and
            # *much* longer (5-15 s) on the very first call after launch
            # while the model was lazy-loading. During that window EVERY
            # other endpoint (status, queue, watch) also stalled, which
            # surfaced to the user as "the UI is frozen and search times
            # out." `embed_text_async` delegates to the shared pipeline
            # executor so other handlers keep running.
            embed_start = time.perf_counter()
            # priority=True jumps the embedder's per-model gate ahead of
            # any indexing-side encode that hasn't started yet, and
            # asyncio.to_thread (used by the async path) keeps us off
            # the parser thread pool so a multi-second markitdown parse
            # can't queue us behind it.
            query_embedding = await self.embedder.embed_text_async(query, priority=True)
            embed_ms = (time.perf_counter() - embed_start) * 1000.0

            db_start = time.perf_counter()
            results = await self.db.search_similar_files(
                query_embedding=query_embedding,
                limit=limit,
                threshold=threshold,
                directory=directory,
                path_pattern=path_pattern,
            )
            db_ms = (time.perf_counter() - db_start) * 1000.0

            logger.info("Semantic search timing",
                        query_len=len(query), embed_ms=round(embed_ms, 1),
                        db_ms=round(db_ms, 1), candidates=len(results))
            return results


        except Exception as e:
            logger.exception("Semantic search failed", error=str(e))
            return []

    async def _keyword_search(self, query: str, limit: int, directory: str | None = None, path_pattern: str | None = None) -> list[tuple]:
        """Perform keyword search using SQLite FTS5."""
        try:
            # Use database FTS5 for efficient keyword search
            results = await self.db.keyword_search(query, limit * 2, directory, path_pattern=path_pattern)

            # Results are already in (file_metadata, score) format
            # FTS5 returns relevance scores that work well for ranking
            return results

        except Exception as e:
            logger.exception("Keyword search failed", error=str(e))
            return []

    async def _apps_search(self, query: str, limit: int, *, path_pattern: str | None = None) -> list[tuple[Application, float]]:
        """Fused keyword + semantic search against the applications source.

        Two queries fire in parallel:
          * applications_fts BM25 (literal-token recall — "Logic Pro",
            "Docker", bundle ids).
          * application_embeddings knn vec search (intent recall —
            "music production" finds Logic Pro even when its
            description doesn't contain those tokens).
        Results are RRF-merged the same way files are. Returned as
        (Application, score) tuples where score is the RRF score
        (higher = better, opposite of raw vec distance) so the
        existing API/UI consumers can sort descending without
        special-casing.
        """
        try:
            # Apps are a small dataset (< 1000 typical), so a smaller
            # limit is fine — we surface the top match-density per
            # category, not exhaustive results.
            kw_task = asyncio.create_task(
                self.db.applications_keyword_search(query, limit, path_pattern=path_pattern)
            )
            # Semantic side: only meaningful when we have an embedder.
            # Without one we'd be issuing zero-vector knn queries that
            # return arbitrary near-zero matches.
            sem_coro: asyncio.Future
            if self.embedder is not None:
                async def _sem() -> list[tuple[Application, float]]:
                    vec = await self.embedder.embed_text_async(query, priority=True)
                    return await self.db.search_similar_applications(vec, limit, path_pattern=path_pattern)
                sem_task = asyncio.create_task(_sem())
            else:
                async def _empty() -> list[tuple[Application, float]]:
                    return []
                sem_task = asyncio.create_task(_empty())

            kw_result, sem_result = await asyncio.gather(
                kw_task, sem_task, return_exceptions=True,
            )

            kw_list: list[tuple[Application, float]] = (
                kw_result if not isinstance(kw_result, Exception) else []
            )
            sem_list: list[tuple[Application, float]] = (
                sem_result if not isinstance(sem_result, Exception) else []
            )
            if isinstance(kw_result, Exception):
                logger.warning("Apps keyword search failed", error=str(kw_result))
            if isinstance(sem_result, Exception):
                logger.warning("Apps semantic search failed", error=str(sem_result))

            # RRF wants (id, data, score) tuples in best-first order.
            # Apps have unique ids — use them directly.
            kw_tuples = [(a.id, a, s) for a, s in kw_list if a.id is not None]
            sem_tuples = [(a.id, a, s) for a, s in sem_list if a.id is not None]
            merged = merge_with_rrf(sem_tuples, kw_tuples)

            # Re-shape into (Application, rrf_score) for the caller.
            return [(data, rrf_score) for _id, data, rrf_score, *_ in merged][:limit]
        except Exception as e:
            logger.exception("Applications fused search failed", error=str(e))
            return []

    async def search_with_apps(
        self,
        query: str,
        limit: int = 50,
        *,
        scope: SearchScope = "all",
        directory: str | None = None,
        path_pattern: str | None = None,
        semantic_threshold: float = 2.0,
        alpha_floor: float = 0.4,
        rrf_k: int = 60,
    ) -> SearchBundle:
        """Unified hybrid search that ranks files AND applications in
        the same RRF, then splits the top hits back into two groups
        for the UI.

        Behavior change vs the previous design: apps are no longer a
        parallel sidecar that always populates. They compete with
        files in one cross-modal ranking — irrelevant apps drop out
        naturally, relevant ones can bubble to the top.

        Args:
            query: User search string.
            limit: Hard cap on total returned hits (files + apps).
            scope: ``"all"`` runs all four modality queries; ``"files"``
                skips the apps branches; ``"applications"`` skips the
                file branches. Set by the frontend via ``@Applications``
                or other scope tokens.
            directory: Optional folder scope for the file branches.
            path_pattern: Optional glob (e.g. ``"*.pdf"``) pushed into
                the SQL WHERE clause as a LIKE filter; applies to both
                files (file_path) and apps (display_name/app_path).
            semantic_threshold: Max vec0 distance for semantic hits.
            alpha_floor: Relative-score cutoff. Items with RRF score
                below ``alpha_floor * top_score`` are dropped — so a
                weak tail can't pad a thin result list. Set to 0 to
                disable.
            rrf_k: RRF smoothing constant.
        """
        t_start = time.perf_counter()
        fetch_limit = max(limit * 3, 30)
        logger.info(
            "Unified search received",
            query=query,
            limit=limit,
            scope=scope,
            directory=directory,
            path_pattern=path_pattern,
            alpha_floor=alpha_floor,
        )

        # Build the modality task set based on scope. Empty branches
        # would be wasted work — scope="applications" should not run
        # the file FTS at all, etc.
        run_files = scope in ("all", "files")
        run_apps = scope in ("all", "applications")

        tasks: dict[str, asyncio.Task] = {}
        if run_files:
            tasks["file_sem"] = asyncio.create_task(
                self._semantic_search(query, fetch_limit, semantic_threshold, directory, path_pattern)
            )
            tasks["file_kw"] = asyncio.create_task(
                self._keyword_search(query, fetch_limit, directory, path_pattern)
            )
        if run_apps:
            # Apps modalities are inlined here rather than via
            # _apps_search so each (semantic, keyword) ranked list
            # contributes to the cross-modal RRF independently. The
            # legacy _apps_search performs its own internal RRF, which
            # would double-fuse if we used it here.
            tasks["app_kw"] = asyncio.create_task(
                self.db.applications_keyword_search(query, fetch_limit, path_pattern=path_pattern)
            )
            if self.embedder is not None:
                async def _app_sem() -> list[tuple[Application, float]]:
                    vec = await self.embedder.embed_text_async(query, priority=True)
                    return await self.db.search_similar_applications(vec, fetch_limit, path_pattern=path_pattern)
                tasks["app_sem"] = asyncio.create_task(_app_sem())

        t_lookup = time.perf_counter()
        raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
        lookup_ms = (time.perf_counter() - t_lookup) * 1000.0
        by_key = dict(zip(tasks.keys(), raw, strict=False))

        # Build namespaced ranked lists feeding the RRF. IDs are
        # tagged ("file", id) / ("app", id) so collisions across the
        # two sources can't happen.
        ranked_lists: list[list[tuple[tuple[str, int], object]]] = []
        file_data: dict[int, File] = {}
        app_data: dict[int, Application] = {}
        per_list_ranks: dict[str, dict[tuple[str, int], int]] = {}

        def _ingest(list_name: str, items: list, kind: str, store: dict) -> None:
            rl: list[tuple[tuple[str, int], object]] = []
            ranks: dict[tuple[str, int], int] = {}
            for rank, pair in enumerate(items):
                obj, _score = pair
                obj_id = getattr(obj, "id", None)
                if obj_id is None:
                    continue
                key = (kind, obj_id)
                rl.append((key, obj))
                store[obj_id] = obj
                ranks[key] = rank
            if rl:
                ranked_lists.append(rl)
                per_list_ranks[list_name] = ranks

        for key in ("file_sem", "file_kw", "app_sem", "app_kw"):
            result = by_key.get(key)
            if result is None:
                continue
            if isinstance(result, Exception):
                logger.warning("Search branch failed", branch=key, error=str(result))
                continue
            kind = "file" if key.startswith("file_") else "app"
            store = file_data if kind == "file" else app_data
            _ingest(key, result, kind, store)

        # One cross-modal RRF over up to 4 namespaced ranked lists.
        rrf_scores = compute_rrf_scores(ranked_lists, k=rrf_k)

        # Sort by RRF score desc, then apply relative threshold +
        # hard cap. The relative threshold (alpha_floor * top) is
        # what makes "always show 8 apps" go away — quiet apps with
        # tiny RRF contributions get dropped instead of padding.
        sorted_hits = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        if sorted_hits and alpha_floor > 0:
            top_score = sorted_hits[0][1]
            cutoff = top_score * alpha_floor
            sorted_hits = [(k, s) for k, s in sorted_hits if s >= cutoff]
        sorted_hits = sorted_hits[:limit]

        files: list[SearchResult] = []
        apps: list[ApplicationResult] = []
        for (kind, obj_id), score in sorted_hits:
            if kind == "file":
                f = file_data.get(obj_id)
                if f is None:
                    continue
                in_sem = (kind, obj_id) in per_list_ranks.get("file_sem", {})
                in_kw = (kind, obj_id) in per_list_ranks.get("file_kw", {})
                if in_sem and in_kw:
                    mtype = "hybrid"
                elif in_sem:
                    mtype = "semantic"
                else:
                    mtype = "keyword"
                files.append(SearchResult(
                    file_metadata=f,
                    combined_score=score,
                    match_type=mtype,
                    semantic_rank=per_list_ranks.get("file_sem", {}).get((kind, obj_id)),
                    keyword_rank=per_list_ranks.get("file_kw", {}).get((kind, obj_id)),
                ))
            else:
                a = app_data.get(obj_id)
                if a is None:
                    continue
                apps.append(ApplicationResult(application=a, relevance_score=score))

        total_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Unified search completed",
            query=query,
            total_ms=round(total_ms, 1),
            lookup_ms=round(lookup_ms, 1),
            files=len(files),
            apps=len(apps),
        )
        return SearchBundle(files=files, apps=apps, total_ms=total_ms)

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
