"""
Search Module

Provides hybrid search combining:
1. Semantic search - Vector similarity using embeddings
2. Full-text search - FTS5 with BM25 ranking

Results are fused using Reciprocal Rank Fusion (RRF) which
combines rankings from both methods without needing score normalization.

Components:
- HybridSearcher: Main search orchestrator
- fts5_query: FTS5 query parsing and sanitization
- rrf: Reciprocal Rank Fusion algorithm
"""

from .searcher import HybridSearcher as HybridSearcher
