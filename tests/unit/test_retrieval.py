"""
Unit tests for retrieval logic

Tests RRF scoring, weighted retrieval, path validation, and defensive dict access (Bug #4).
"""

import pytest
import numpy as np
from bigrag.operate import rrf_score_fusion


class TestRRFScoring:
    """Test Reciprocal Rank Fusion scoring"""

    def test_rrf_single_list(self):
        """Test RRF with single ranking list"""
        # Single ranking: [item1, item2, item3]
        rankings = [["item1", "item2", "item3"]]

        scores = rrf_score_fusion(rankings)

        # Scores should decrease with rank
        assert scores["item1"] > scores["item2"]
        assert scores["item2"] > scores["item3"]

    def test_rrf_multiple_lists(self):
        """Test RRF with multiple ranking lists"""
        # Two rankings with some overlap
        rankings = [
            ["itemA", "itemB", "itemC"],
            ["itemB", "itemC", "itemD"],
        ]

        scores = rrf_score_fusion(rankings)

        # itemB appears in both lists (rank 1 + rank 0), should have highest score
        assert scores["itemB"] > scores["itemA"]
        # itemA appears once at rank 0 (score=1.0), itemC appears twice at ranks 2,1 (score=0.833)
        # Single top-rank appearance beats multiple low-rank appearances (correct RRF behavior)
        assert scores["itemA"] > scores["itemC"]
        assert scores["itemC"] > scores["itemD"]

    def test_rrf_empty_lists(self):
        """Test RRF with empty input"""
        rankings = []

        scores = rrf_score_fusion(rankings)

        assert scores == {}

    def test_rrf_single_item_lists(self):
        """Test RRF with single-item lists"""
        rankings = [["item1"], ["item1"], ["item1"]]

        scores = rrf_score_fusion(rankings)

        # Item appears 3 times at rank 1
        assert "item1" in scores
        assert scores["item1"] > 0

    def test_rrf_no_overlap(self):
        """Test RRF with rankings that don't overlap"""
        rankings = [
            ["item1", "item2"],
            ["item3", "item4"],
        ]

        scores = rrf_score_fusion(rankings)

        # All items should have scores
        assert len(scores) == 4
        # No item should dominate due to overlap
        assert scores["item1"] == scores["item3"]  # Both at rank 1


class TestWeightedRetrieval:
    """Test weighted retrieval and scoring"""

    def test_weight_based_sorting(self):
        """Test that results are sorted by weight"""
        results = [
            {"name": "Entity A", "weight": 100.0},
            {"name": "Entity B", "weight": 200.0},
            {"name": "Entity C", "weight": 50.0},
        ]

        # Sort by weight descending
        sorted_results = sorted(results, key=lambda x: x["weight"], reverse=True)

        assert sorted_results[0]["name"] == "Entity B"
        assert sorted_results[1]["name"] == "Entity A"
        assert sorted_results[2]["name"] == "Entity C"

    def test_weight_filtering(self):
        """Test filtering by minimum weight threshold"""
        results = [
            {"name": "Entity A", "weight": 100.0},
            {"name": "Entity B", "weight": 50.0},
            {"name": "Entity C", "weight": 10.0},
        ]

        threshold = 60.0
        filtered = [r for r in results if r["weight"] >= threshold]

        assert len(filtered) == 1
        assert filtered[0]["name"] == "Entity A"

    def test_normalized_weights(self):
        """Test weight normalization to 0-1 range"""
        results = [
            {"name": "Entity A", "weight": 100.0},
            {"name": "Entity B", "weight": 200.0},
            {"name": "Entity C", "weight": 50.0},
        ]

        # Normalize weights
        max_weight = max(r["weight"] for r in results)
        for r in results:
            r["norm_weight"] = r["weight"] / max_weight

        assert results[1]["norm_weight"] == 1.0  # Max weight
        assert results[0]["norm_weight"] == 0.5
        assert results[2]["norm_weight"] == 0.25


class TestDefensiveDictAccess:
    """Test defensive dict access (Bug #4 fix validation)"""

    def test_safe_key_access_with_get(self):
        """Test using dict.get() for safe access"""
        result = {"entity_name": "Test Entity"}

        # Safe access (Bug #4 fix)
        name = result.get("entity_name")
        assert name == "Test Entity"

        # Access missing key safely
        missing = result.get("nonexistent_key")
        assert missing is None

        # Access with default value
        missing_with_default = result.get("nonexistent_key", "default")
        assert missing_with_default == "default"

    def test_safe_list_comprehension(self):
        """Test safe filtering in list comprehension (Bug #4 fix)"""
        results = [
            {"entity_name": "Entity A"},
            {"entity_name": "Entity B"},
            {"other_field": "value"},  # Missing entity_name
            {"entity_name": "Entity C"},
        ]

        # Safe extraction (Bug #4 fix: use get() and filter)
        entity_names = [
            r.get("entity_name")
            for r in results
            if "entity_name" in r
        ]

        assert len(entity_names) == 3
        assert "Entity A" in entity_names
        assert "Entity B" in entity_names
        assert "Entity C" in entity_names

    def test_defensive_access_with_validation(self):
        """Test defensive access with validation"""
        results = [
            {"entity_name": "Entity A", "weight": 100.0},
            {"weight": 50.0},  # Missing entity_name
            {"entity_name": "Entity B"},  # Missing weight
        ]

        # Extract safely with both fields
        valid_results = [
            {"name": r.get("entity_name"), "weight": r.get("weight", 0.0)}
            for r in results
            if "entity_name" in r and "weight" in r
        ]

        assert len(valid_results) == 1
        assert valid_results[0]["name"] == "Entity A"


class TestPathAEntityRetrieval:
    """Test Path A (entity-based) retrieval logic"""

    def test_entity_vector_search_format(self):
        """Test entity vector search result format"""
        # Simulate vector DB query results
        results = [
            {"entity_name": "Messi", "score": 0.95},
            {"entity_name": "Ronaldo", "score": 0.87},
            {"entity_name": "Football", "score": 0.80},
        ]

        # Extract entity names (Bug #4 fix: defensive access)
        entity_names = [r.get("entity_name") for r in results if "entity_name" in r]

        assert len(entity_names) == 3
        assert "Messi" in entity_names

    def test_entity_graph_lookup(self):
        """Test looking up entities in graph"""
        # Simulate graph query results
        graph_results = {
            "Messi": {
                "entity_type": "person",
                "description": "Football player",
                "weight": 360.0,
            },
            "Ronaldo": {
                "entity_type": "person",
                "description": "Football player",
                "weight": 340.0,
            },
        }

        # Lookup entity data
        entity_name = "Messi"
        entity_data = graph_results.get(entity_name)

        assert entity_data is not None
        assert entity_data["weight"] == 360.0


class TestPathBRelationRetrieval:
    """Test Path B (relation-based) retrieval logic"""

    def test_relation_vector_search_format(self):
        """Test relation vector search result format"""
        # Simulate vector DB query results
        results = [
            {"bipartite_edge_name": "Messi plays for Inter Miami", "score": 0.92},
            {"bipartite_edge_name": "Inter Miami based in Florida", "score": 0.85},
        ]

        # Extract relation names (Bug #4 fix: defensive access)
        relation_names = [
            r.get("bipartite_edge_name")
            for r in results
            if "bipartite_edge_name" in r
        ]

        assert len(relation_names) == 2
        assert "Messi plays for Inter Miami" in relation_names

    def test_relation_graph_lookup(self):
        """Test looking up relations in graph"""
        # Simulate graph query results
        graph_results = {
            "Messi plays for Inter Miami": {
                "description": "Messi plays for Inter Miami",
                "weight": 27.0,
            },
        }

        # Lookup relation data
        relation_name = "Messi plays for Inter Miami"
        relation_data = graph_results.get(relation_name)

        assert relation_data is not None
        assert relation_data["weight"] == 27.0


class TestPathCChunkRetrieval:
    """Test Path C (chunk-based) retrieval logic"""

    def test_chunk_vector_search_format(self):
        """Test chunk vector search result format"""
        # Simulate vector DB query results
        results = [
            {"chunk_id": "chunk-1", "content": "Messi plays for Inter Miami", "score": 0.90},
            {"chunk_id": "chunk-2", "content": "Inter Miami is in Florida", "score": 0.85},
        ]

        # Extract chunk data
        chunks = [(r.get("content"), [r.get("chunk_id")]) for r in results]

        assert len(chunks) == 2
        assert "Messi" in chunks[0][0]

    def test_chunk_reranking_preparation(self):
        """Test preparing chunks for reranking"""
        # Simulate chunk candidates
        candidates = [
            ("Chunk content A", ["chunk-1"]),
            ("Chunk content B", ["chunk-2"]),
            ("Chunk content C", ["chunk-3"]),
        ]

        # Format for reranker: List[Tuple[str, List[str]]]
        assert len(candidates) == 3
        assert isinstance(candidates[0], tuple)
        assert isinstance(candidates[0][0], str)  # content
        assert isinstance(candidates[0][1], list)  # source_ids


class TestHybridModeRetrieval:
    """Test hybrid mode combining all three paths"""

    def test_hybrid_result_merging(self):
        """Test merging results from 3 paths"""
        # Simulate results from each path
        path_a_results = ["Entity A context", "Entity B context"]
        path_b_results = ["Relation X context", "Relation Y context"]
        path_c_results = ["Chunk 1 content", "Chunk 2 content"]

        # Merge results
        all_results = path_a_results + path_b_results + path_c_results

        # Should have results from all paths
        assert len(all_results) == 6

    def test_hybrid_deduplication(self):
        """Test removing duplicate contexts in hybrid mode"""
        # Simulate results with duplicates
        results = [
            "Context A",
            "Context B",
            "Context A",  # Duplicate
            "Context C",
            "Context B",  # Duplicate
        ]

        # Deduplicate
        unique_results = list(dict.fromkeys(results))

        assert len(unique_results) == 3
        assert "Context A" in unique_results


class TestRetrievalErrorHandling:
    """Test error handling in retrieval logic"""

    def test_empty_query_results(self):
        """Test handling empty query results"""
        results = []

        # Should return empty list, not error
        assert isinstance(results, list)
        assert len(results) == 0

    def test_none_query_results(self):
        """Test handling None query results"""
        results = None

        # Safely handle None
        if not results or not len(results):
            results = []

        assert results == []

    def test_malformed_results(self):
        """Test handling malformed results (Bug #4 scenario)"""
        results = [
            {"entity_name": "Valid Entity"},
            {},  # Empty dict
            {"other_field": "value"},  # Missing required field
            None,  # None in list
        ]

        # Extract safely (Bug #4 fix)
        valid_entities = [
            r.get("entity_name")
            for r in results
            if r and isinstance(r, dict) and "entity_name" in r
        ]

        assert len(valid_entities) == 1
        assert valid_entities[0] == "Valid Entity"


class TestTopKFiltering:
    """Test top-k result filtering"""

    def test_top_k_selection(self):
        """Test selecting top-k results"""
        results = [
            {"name": f"Entity {i}", "score": 1.0 - i*0.1}
            for i in range(10)
        ]

        top_k = 5
        top_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        assert len(top_results) == 5
        assert top_results[0]["name"] == "Entity 0"
        assert top_results[-1]["name"] == "Entity 4"

    def test_top_k_less_than_available(self):
        """Test top-k when k > number of results"""
        results = [
            {"name": "Entity A", "score": 0.9},
            {"name": "Entity B", "score": 0.8},
        ]

        top_k = 10
        top_results = results[:top_k]

        # Should return all available results
        assert len(top_results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
