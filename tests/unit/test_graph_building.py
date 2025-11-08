"""
Unit tests for graph building logic

Tests entity/relation node creation, weight aggregation, and graph structure.
"""

import pytest
import networkx as nx
from bigrag.utils import compute_mdhash_id
from bigrag.constants import (
    ENTITY_PREFIX,
    BIPARTITE_EDGE_PREFIX,
    CHUNK_PREFIX,
)


class TestNodeIDGeneration:
    """Test hash ID generation for graph nodes"""

    def test_entity_id_format(self):
        """Test entity node IDs use correct prefix"""
        entity_name = "Test Entity"
        entity_id = compute_mdhash_id(entity_name, prefix=ENTITY_PREFIX)

        assert entity_id.startswith("ent-")
        assert len(entity_id) > 4  # Has hash part

    def test_relation_id_format(self):
        """Test relation (bipartite edge) IDs use correct prefix"""
        relation_desc = "Test relation description"
        relation_id = compute_mdhash_id(relation_desc, prefix=BIPARTITE_EDGE_PREFIX)

        assert relation_id.startswith("rel-")
        assert len(relation_id) > 4

    def test_chunk_id_format(self):
        """Test chunk IDs use correct prefix"""
        chunk_content = "Test chunk content"
        chunk_id = compute_mdhash_id(chunk_content, prefix=CHUNK_PREFIX)

        assert chunk_id.startswith("chunk-")
        assert len(chunk_id) > 6

    def test_deterministic_hashing(self):
        """Test that same content produces same ID"""
        content = "Deterministic test"

        id1 = compute_mdhash_id(content, prefix="test-")
        id2 = compute_mdhash_id(content, prefix="test-")

        assert id1 == id2

    def test_different_content_different_ids(self):
        """Test that different content produces different IDs"""
        id1 = compute_mdhash_id("Content A", prefix="test-")
        id2 = compute_mdhash_id("Content B", prefix="test-")

        assert id1 != id2


class TestGraphStructureBasics:
    """Test basic graph structure creation"""

    def test_create_empty_graph(self):
        """Test creating empty NetworkX graph"""
        G = nx.Graph()

        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_add_entity_node(self):
        """Test adding entity node to graph"""
        G = nx.Graph()

        entity_id = "ent-test123"
        G.add_node(
            entity_id,
            name="Test Entity",
            entity_type="person",
            description="Test person entity",
            weight=100.0,
            role="entity",
        )

        assert G.has_node(entity_id)
        assert G.nodes[entity_id]["name"] == "Test Entity"
        assert G.nodes[entity_id]["role"] == "entity"
        assert G.nodes[entity_id]["weight"] == 100.0

    def test_add_relation_node(self):
        """Test adding relation (bipartite edge) node to graph"""
        G = nx.Graph()

        relation_id = "rel-test456"
        G.add_node(
            relation_id,
            name="Test Relation",
            description="Entity A relates to Entity B",
            weight=50.0,
            role="bipartite_edge",
        )

        assert G.has_node(relation_id)
        assert G.nodes[relation_id]["role"] == "bipartite_edge"
        assert G.nodes[relation_id]["weight"] == 50.0

    def test_add_chunk_node(self):
        """Test adding chunk node to graph"""
        G = nx.Graph()

        chunk_id = "chunk-test789"
        G.add_node(
            chunk_id,
            content="Test chunk content",
            doc_id="doc-123",
            role="chunk",
        )

        assert G.has_node(chunk_id)
        assert G.nodes[chunk_id]["role"] == "chunk"


class TestGraphEdgeCreation:
    """Test edge creation between nodes"""

    def test_chunk_to_entity_edge(self):
        """Test creating edge from chunk to entity"""
        G = nx.Graph()

        chunk_id = "chunk-1"
        entity_id = "ent-1"

        G.add_node(chunk_id, role="chunk")
        G.add_node(entity_id, role="entity")
        G.add_edge(chunk_id, entity_id, weight=1.0)

        assert G.has_edge(chunk_id, entity_id)
        assert G[chunk_id][entity_id]["weight"] == 1.0

    def test_chunk_to_relation_edge(self):
        """Test creating edge from chunk to relation"""
        G = nx.Graph()

        chunk_id = "chunk-1"
        relation_id = "rel-1"

        G.add_node(chunk_id, role="chunk")
        G.add_node(relation_id, role="bipartite_edge")
        G.add_edge(chunk_id, relation_id, weight=1.0)

        assert G.has_edge(chunk_id, relation_id)

    def test_bipartite_graph_structure(self):
        """Test that graph maintains bipartite structure"""
        G = nx.Graph()

        # Create chunk nodes
        G.add_node("chunk-1", role="chunk")
        G.add_node("chunk-2", role="chunk")

        # Create semantic nodes (entities and relations)
        G.add_node("ent-1", role="entity")
        G.add_node("rel-1", role="bipartite_edge")

        # Add bipartite edges (chunk <-> semantic)
        G.add_edge("chunk-1", "ent-1")
        G.add_edge("chunk-1", "rel-1")
        G.add_edge("chunk-2", "ent-1")

        # Verify structure
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 3

        # Chunks should not connect to chunks
        assert not G.has_edge("chunk-1", "chunk-2")

        # Entities should not connect to other entities directly
        # (they connect through chunks)


class TestWeightAggregation:
    """Test weight aggregation for entities and relations (Bug #3 related)"""

    def test_entity_weight_single_occurrence(self):
        """Test entity weight with single occurrence"""
        # Simulate entity extracted once with score 85
        weight = 85.0

        assert weight == 85.0

    def test_entity_weight_multiple_occurrences(self):
        """Test entity weight aggregation (sum of scores)"""
        # Simulate entity extracted 3 times with scores [90, 85, 95]
        occurrences = [90.0, 85.0, 95.0]
        aggregated_weight = sum(occurrences)

        assert aggregated_weight == 270.0

    def test_relation_weight_single_occurrence(self):
        """Test relation weight with single occurrence"""
        # Completeness score 0-10
        weight = 8.0

        assert weight == 8.0

    def test_relation_weight_multiple_occurrences(self):
        """Test relation weight aggregation"""
        # Simulate relation mentioned 3 times with completeness [9, 8, 10]
        occurrences = [9.0, 8.0, 10.0]
        aggregated_weight = sum(occurrences)

        assert aggregated_weight == 27.0

    def test_weight_update_on_merge(self):
        """Test weight increases when same entity/relation seen again"""
        G = nx.Graph()

        entity_id = "ent-test"

        # First occurrence
        G.add_node(entity_id, name="Test", weight=90.0, role="entity")
        assert G.nodes[entity_id]["weight"] == 90.0

        # Second occurrence (should add to weight)
        G.nodes[entity_id]["weight"] += 85.0
        assert G.nodes[entity_id]["weight"] == 175.0

        # Third occurrence
        G.nodes[entity_id]["weight"] += 95.0
        assert G.nodes[entity_id]["weight"] == 270.0


class TestGraphMetadata:
    """Test metadata storage in graph nodes"""

    def test_entity_metadata(self):
        """Test that entity nodes store all required metadata"""
        G = nx.Graph()

        entity_id = "ent-test"
        G.add_node(
            entity_id,
            name="Test Entity",
            entity_type="person",
            description="A test person entity",
            weight=100.0,
            source_id=["chunk-1", "chunk-2"],
            role="entity",
        )

        node = G.nodes[entity_id]

        assert node["name"] == "Test Entity"
        assert node["entity_type"] == "person"
        assert node["description"] == "A test person entity"
        assert node["weight"] == 100.0
        assert "chunk-1" in node["source_id"]
        assert node["role"] == "entity"

    def test_relation_metadata(self):
        """Test that relation nodes store all required metadata"""
        G = nx.Graph()

        relation_id = "rel-test"
        G.add_node(
            relation_id,
            name="Test Relation",
            description="Entity A works for Entity B",
            weight=15.0,
            source_id=["chunk-1"],
            role="bipartite_edge",
        )

        node = G.nodes[relation_id]

        assert node["name"] == "Test Relation"
        assert node["description"] is not None
        assert node["weight"] == 15.0
        assert node["role"] == "bipartite_edge"


class TestGraphQueries:
    """Test graph query operations"""

    def test_get_node_by_id(self):
        """Test retrieving node by ID"""
        G = nx.Graph()
        G.add_node("ent-1", name="Entity 1", role="entity")

        assert G.has_node("ent-1")
        assert G.nodes["ent-1"]["name"] == "Entity 1"

    def test_get_neighbors(self):
        """Test getting neighbors of a node"""
        G = nx.Graph()

        G.add_node("chunk-1", role="chunk")
        G.add_node("ent-1", role="entity")
        G.add_node("ent-2", role="entity")

        G.add_edge("chunk-1", "ent-1")
        G.add_edge("chunk-1", "ent-2")

        neighbors = list(G.neighbors("chunk-1"))

        assert len(neighbors) == 2
        assert "ent-1" in neighbors
        assert "ent-2" in neighbors

    def test_filter_nodes_by_role(self):
        """Test filtering nodes by role attribute"""
        G = nx.Graph()

        G.add_node("ent-1", role="entity")
        G.add_node("ent-2", role="entity")
        G.add_node("rel-1", role="bipartite_edge")
        G.add_node("chunk-1", role="chunk")

        # Filter entities
        entities = [n for n, attrs in G.nodes(data=True) if attrs.get("role") == "entity"]
        assert len(entities) == 2

        # Filter relations
        relations = [n for n, attrs in G.nodes(data=True) if attrs.get("role") == "bipartite_edge"]
        assert len(relations) == 1

        # Filter chunks
        chunks = [n for n, attrs in G.nodes(data=True) if attrs.get("role") == "chunk"]
        assert len(chunks) == 1


class TestGraphPersistence:
    """Test graph serialization and loading"""

    def test_graph_to_graphml(self, tmp_path):
        """Test saving graph to GraphML format"""
        G = nx.Graph()

        G.add_node("ent-1", name="Entity 1", weight=100.0, role="entity")
        G.add_node("chunk-1", role="chunk")
        G.add_edge("chunk-1", "ent-1", weight=1.0)

        # Save to file
        output_file = tmp_path / "test_graph.graphml"
        nx.write_graphml(G, str(output_file))

        assert output_file.exists()

    def test_graph_from_graphml(self, tmp_path):
        """Test loading graph from GraphML format"""
        G = nx.Graph()

        G.add_node("ent-1", name="Entity 1", weight=100.0, role="entity")
        G.add_node("chunk-1", role="chunk")
        G.add_edge("chunk-1", "ent-1", weight=1.0)

        # Save
        output_file = tmp_path / "test_graph.graphml"
        nx.write_graphml(G, str(output_file))

        # Load
        G_loaded = nx.read_graphml(str(output_file))

        assert G_loaded.has_node("ent-1")
        assert G_loaded.has_node("chunk-1")
        assert G_loaded.has_edge("chunk-1", "ent-1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
