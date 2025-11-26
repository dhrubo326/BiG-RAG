"""
Bipartite Graph Builder for Production Knowledge Graph

Converts ProductionKGPipeline output to BiG-RAG graph structure:
- Entities → V_E partition (entity nodes)
- Relations → V_R partition (relation nodes)
- Bipartite edges: V_R ↔ V_E

Integration with BiG-RAG:
- Compatible with NetworkX, Neo4j, and other graph storage backends
- Preserves all metadata from extraction pipeline
- Creates proper bipartite structure for three-path retrieval
"""

import asyncio
from typing import Dict, List, Optional
from bigrag.utils import compute_mdhash_id, logger
from bigrag.base import BaseGraphStorage, BaseVectorStorage
from bigrag.constants import RELATION_PREFIX


class BipartiteGraphBuilder:
    """
    Convert ProductionKGPipeline extraction results to BiG-RAG bipartite graph.

    Input: Pipeline result from ProductionKGPipeline.process_document()
    Output: BiG-RAG graph with nodes and edges upserted to storage

    Graph Structure:
    - V_E (entity nodes): Individual entities with descriptions
    - V_R (relation nodes): Knowledge segments (from table rows or paragraphs)
    - Edges: relation_id → entity_name (bipartite edges)

    Usage:
        builder = BipartiteGraphBuilder()
        stats = await builder.build_graph(
            pipeline_result,
            rag.chunk_entity_relation_graph,
            rag.vdb_entities,
            rag.vdb_relations
        )
    """

    async def build_graph(
        self,
        pipeline_result: Dict,
        knowledge_graph_inst: BaseGraphStorage,
        vdb_entities: BaseVectorStorage,
        vdb_relations: BaseVectorStorage,
    ) -> Dict:
        """
        Build bipartite graph from pipeline extraction results.

        Args:
            pipeline_result: Output from ProductionKGPipeline.process_document()
                Expected keys: 'entities', 'relations', 'chunks', 'validation', 'statistics'
            knowledge_graph_inst: Graph storage (NetworkX, Neo4j, etc.)
            vdb_entities: Vector DB for entity embeddings (Path A retrieval)
            vdb_relations: Vector DB for relation embeddings (Path B retrieval)

        Returns:
            Statistics dict:
            {
                'entity_nodes': int,
                'relation_nodes': int,
                'bipartite_edges': int,
                'orphan_relations': int  # Relations with no linked entities
            }
        """

        entities = pipeline_result.get('entities', [])
        relations = pipeline_result.get('relations', [])

        logger.info(f"[GraphBuilder] ========== GRAPH BUILDER START ==========")
        logger.info(f"[GraphBuilder] Received pipeline_result with keys: {list(pipeline_result.keys())}")
        logger.info(f"[GraphBuilder] Building graph from {len(entities)} entities, {len(relations)} relations")

        if entities:
            logger.info(f"[GraphBuilder] Sample entities (first 3):")
            for i, entity in enumerate(entities[:3]):
                logger.info(f"  Entity {i+1}: name='{entity.get('entity_name', 'NO_NAME')}', type={entity.get('entity_type', 'NO_TYPE')}, has_entity_id={bool(entity.get('entity_id'))}")
        else:
            logger.warning(f"[GraphBuilder] NO ENTITIES in pipeline_result!")

        if relations:
            logger.info(f"[GraphBuilder] Sample relations (first 3):")
            for i, relation in enumerate(relations[:3]):
                content_preview = relation.get('content', 'NO_CONTENT')[:60]
                linked_entities = relation.get('metadata', {}).get('linked_entities', [])
                logger.info(f"  Relation {i+1}: content='{content_preview}...', linked_entities_count={len(linked_entities)}")
        else:
            logger.warning(f"[GraphBuilder] NO RELATIONS in pipeline_result!")

        # Step 1: Create relation nodes (V_R partition)
        logger.info(f"[GraphBuilder] STEP 1: Creating {len(relations)} relation nodes...")
        relation_nodes_created = await self._create_relation_nodes(
            relations,
            knowledge_graph_inst,
            vdb_relations
        )
        logger.info(f"[GraphBuilder] STEP 1 Complete: {relation_nodes_created} relation nodes created")

        # Step 2: Create entity nodes (V_E partition)
        logger.info(f"[GraphBuilder] STEP 2: Creating {len(entities)} entity nodes...")
        entity_nodes_created = await self._create_entity_nodes(
            entities,
            knowledge_graph_inst,
            vdb_entities
        )
        logger.info(f"[GraphBuilder] STEP 2 Complete: {entity_nodes_created} entity nodes created")

        # Step 3: Create bipartite edges (V_R → V_E)
        logger.info(f"[GraphBuilder] STEP 3: Creating bipartite edges...")
        edges_created, orphan_count = await self._create_bipartite_edges(
            relations,
            knowledge_graph_inst
        )
        logger.info(f"[GraphBuilder] STEP 3 Complete: {edges_created} edges created, {orphan_count} orphans")

        logger.info(
            f"[GraphBuilder] ========== SUMMARY ==========\n"
            f"  Entity nodes: {entity_nodes_created}\n"
            f"  Relation nodes: {relation_nodes_created}\n"
            f"  Bipartite edges: {edges_created}\n"
            f"  Orphan relations: {orphan_count}"
        )

        if orphan_count > 0:
            logger.warning(
                f"[GraphBuilder] Found {orphan_count} orphan relations (no linked entities). "
                f"This may indicate extraction issues."
            )

        return {
            'entity_nodes': entity_nodes_created,
            'relation_nodes': relation_nodes_created,
            'bipartite_edges': edges_created,
            'orphan_relations': orphan_count
        }

    async def _create_relation_nodes(
        self,
        relations: List[Dict],
        graph: BaseGraphStorage,
        vdb: BaseVectorStorage
    ) -> int:
        """
        Create relation nodes in V_R partition.

        Each relation becomes a node in the bipartite graph.
        Relations are indexed to vector DB for Path B (relation-based) retrieval.

        Args:
            relations: List of relation dicts from pipeline
            graph: Graph storage backend
            vdb: Vector DB for relation embeddings

        Returns:
            Number of relation nodes created
        """

        count = 0
        for relation in relations:
            # Generate relation ID (hash of content for uniqueness)
            # UNIFIED: Use RELATION_PREFIX constant (rel-) for compatibility with standard pipeline
            relation_id = compute_mdhash_id(relation['content'], prefix=RELATION_PREFIX)

            # Create node data (compatible with BiG-RAG storage format)
            node_data = {
                'role': 'relation',
                'content': relation['content'],
                'description': relation.get('description', relation['content']),  # Fallback to content
                'weight': relation.get('completeness_score', 10),
                'source_id': relation['source_id'],
                'extraction_quality': relation.get('metadata', {}).get('extraction_quality', 'PASS'),  # Track quality level
            }

            # Upsert to graph storage
            await graph.upsert_node(relation_id, node_data=node_data)

            # Upsert to vector DB for semantic search (Path B)
            await vdb.upsert({
                relation_id: {'content': relation['content']}
            })

            count += 1

        return count

    async def _create_entity_nodes(
        self,
        entities: List[Dict],
        graph: BaseGraphStorage,
        vdb: BaseVectorStorage
    ) -> int:
        """
        Create entity nodes in V_E partition.

        Each entity becomes a node in the bipartite graph.
        Entities are indexed to vector DB for Path A (entity-based) retrieval.

        Args:
            entities: List of entity dicts from pipeline
            graph: Graph storage backend
            vdb: Vector DB for entity embeddings

        Returns:
            Number of entity nodes created
        """

        count = 0
        for entity in entities:
            # Option B3: Use entity_id as node ID (stable across name changes)
            entity_id = entity.get('entity_id')
            entity_name = entity['entity_name']

            # Fallback: Use normalized name if no entity_id (backward compatibility)
            if not entity_id:
                entity_id = f'"{entity_name.upper()}"'

            # Create node data (compatible with BiG-RAG storage format)
            # Note: After entity merging, source_id may be a list or missing
            source_id = entity.get('source_id', entity.get('source_ids', ['unknown']))
            if isinstance(source_id, list):
                source_id = source_id[0] if source_id else 'unknown'

            node_data = {
                'role': 'entity',
                'entity_name': entity_name,  # Store name for display
                'entity_type': entity['entity_type'],
                'description': entity['description'],
                'weight': entity.get('weight', 0.0),
                'source_id': source_id,
                'extraction_quality': entity.get('metadata', {}).get('extraction_quality', 'PASS'),  # Track quality level
            }

            # Upsert to graph storage using entity_id
            await graph.upsert_node(entity_id, node_data=node_data)

            # Upsert to vector DB for semantic search (Path A)
            # Store both entity_id and entity_name for retrieval
            await vdb.upsert({
                entity_id: {
                    'content': entity['description'],
                    'entity_id': entity_id,
                    'entity_name': entity_name
                }
            })

            count += 1

        return count

    async def _create_bipartite_edges(
        self,
        relations: List[Dict],
        graph: BaseGraphStorage
    ) -> tuple[int, int]:
        """
        Create bipartite edges connecting relations to entities.

        Bipartite property: Only edges between V_R (relations) and V_E (entities).
        No entity-entity or relation-relation edges.

        Args:
            relations: List of relation dicts with 'linked_entities' in metadata
            graph: Graph storage backend

        Returns:
            (edges_created, orphan_count)
        """

        edges_created = 0
        orphan_count = 0

        for relation in relations:
            # UNIFIED: Use RELATION_PREFIX constant (rel-) for compatibility with standard pipeline
            relation_id = compute_mdhash_id(relation['content'], prefix=RELATION_PREFIX)

            # Option B3: Extract linked entity IDs from metadata (now stores IDs, not names)
            linked_entity_ids = relation.get('metadata', {}).get('linked_entities', [])

            if not linked_entity_ids:
                orphan_count += 1
                logger.warning(
                    f"[GraphBuilder] Relation has no linked entities: "
                    f"'{relation['content'][:80]}...'"
                )
                continue

            # Create edge from relation to each entity
            for entity_id in linked_entity_ids:
                # Option B3: Use entity_id directly (no transformation needed)
                # Entity IDs are stable and already correct after remapping

                # Create bipartite edge (relation → entity)
                edge_data = {
                    'weight': relation.get('completeness_score', 10),
                    'source_id': relation['source_id'],
                }

                await graph.upsert_edge(
                    relation_id,  # Source: relation node (V_R)
                    entity_id,  # Target: entity node (V_E) using stable ID
                    edge_data=edge_data
                )

                edges_created += 1

        return edges_created, orphan_count


async def build_bipartite_graph_from_pipeline(
    pipeline_result: Dict,
    knowledge_graph_inst: BaseGraphStorage,
    vdb_entities: BaseVectorStorage,
    vdb_relations: BaseVectorStorage,
) -> Dict:
    """
    Convenience function to build graph from pipeline result.

    This is the main entry point for converting ProductionKGPipeline output
    to BiG-RAG bipartite graph structure.

    Args:
        pipeline_result: Output from ProductionKGPipeline.process_document()
        knowledge_graph_inst: Graph storage instance
        vdb_entities: Entity vector DB instance
        vdb_relations: Relation vector DB instance

    Returns:
        Statistics dict with node/edge counts

    Example:
        from bigrag import BiGRAG
        from bigrag.production_pipeline import ProductionKGPipeline
        from bigrag.builders import build_bipartite_graph_from_pipeline

        rag = BiGRAG(working_dir="./expr/educational_kg")
        pipeline = ProductionKGPipeline(api_key="your-key")

        result = await pipeline.process_document(doc_text, metadata)
        stats = await build_bipartite_graph_from_pipeline(
            result,
            rag.chunk_entity_relation_graph,
            rag.vdb_entities,
            rag.vdb_relations
        )

        print(f"Created {stats['entity_nodes']} entities, {stats['relation_nodes']} relations")
    """
    builder = BipartiteGraphBuilder()
    stats = await builder.build_graph(
        pipeline_result,
        knowledge_graph_inst,
        vdb_entities,
        vdb_relations
    )
    return stats
