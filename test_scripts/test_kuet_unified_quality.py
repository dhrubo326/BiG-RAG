"""
Comprehensive Quality Test for kuet_unified Knowledge Graph

Tests all aspects of the knowledge graph built from KUET_Admission_info.md:
1. Graph structure and statistics
2. Chunk indexing quality
3. Entity extraction quality
4. Relation extraction quality
5. Three-path retrieval accuracy
6. Question answering capability

Usage:
    python test_kuet_unified_quality.py
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigrag import BiGRAG
from bigrag.base import QueryParam
from bigrag.storage import NetworkXStorage


class KUETGraphQualityTester:
    """Comprehensive quality tester for KUET unified graph"""

    def __init__(self, dataset_name: str = "kuet_unified"):
        self.dataset_name = dataset_name
        # Use absolute path from project root
        project_root = Path(__file__).parent.parent
        self.expr_path = project_root / "expr" / dataset_name
        self.rag = None
        self.test_results = {}

    async def initialize(self):
        """Initialize BiGRAG instance"""
        print(f"\n[INIT] Initializing BiGRAG for dataset: {self.dataset_name}")

        # Check if graph files exist
        graph_file = self.expr_path / "graph_chunk_entity_relation.graphml"
        if not graph_file.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_file}")

        print(f"[INIT] Found graph file: {graph_file}")

        self.rag = BiGRAG(
            working_dir=str(self.expr_path),
        )

        # No need to insert - graph should auto-load
        print("[INIT] BiGRAG initialized successfully")

    async def test_graph_structure(self) -> Dict:
        """Test 1: Graph structure and basic statistics"""
        print("\n" + "="*80)
        print("TEST 1: GRAPH STRUCTURE AND STATISTICS")
        print("="*80)

        results = {}

        # Get graph instance
        graph_storage = self.rag.chunk_entity_relation_graph

        if not isinstance(graph_storage, NetworkXStorage):
            print("[ERROR] Graph storage is not NetworkXStorage!")
            return {"error": "Invalid storage type"}

        graph = graph_storage._graph

        # Basic metrics
        total_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()

        results['total_nodes'] = total_nodes
        results['total_edges'] = total_edges

        print(f"[STATS] Total nodes: {total_nodes}")
        print(f"[STATS] Total edges: {total_edges}")

        # Count nodes by role
        entity_nodes = 0
        relation_nodes = 0
        chunk_nodes = 0
        unknown_nodes = 0

        for node_id, node_data in graph.nodes(data=True):
            role = node_data.get('role', 'unknown')
            if role == 'entity':
                entity_nodes += 1
            elif role == 'relation':
                relation_nodes += 1
            elif role == 'chunk':
                chunk_nodes += 1
            else:
                unknown_nodes += 1

        results['entity_nodes'] = entity_nodes
        results['relation_nodes'] = relation_nodes
        results['chunk_nodes'] = chunk_nodes
        results['unknown_nodes'] = unknown_nodes

        print(f"[STATS] Entity nodes: {entity_nodes}")
        print(f"[STATS] Relation nodes: {relation_nodes}")
        print(f"[STATS] Chunk nodes: {chunk_nodes}")
        print(f"[STATS] Unknown role nodes: {unknown_nodes}")

        # Check for orphan nodes (nodes with no edges)
        orphan_entities = []
        orphan_relations = []

        for node_id, node_data in graph.nodes(data=True):
            if graph.degree(node_id) == 0:
                role = node_data.get('role', 'unknown')
                if role == 'entity':
                    orphan_entities.append((node_id, node_data.get('entity_name', 'NO_NAME')))
                elif role == 'relation':
                    orphan_relations.append((node_id, node_data.get('content', 'NO_CONTENT')[:50]))

        results['orphan_entities'] = len(orphan_entities)
        results['orphan_relations'] = len(orphan_relations)

        print(f"[QUALITY] Orphan entities: {len(orphan_entities)}")
        print(f"[QUALITY] Orphan relations: {len(orphan_relations)}")

        if orphan_entities:
            print(f"[WARNING] Found {len(orphan_entities)} orphan entities (first 5):")
            for node_id, name in orphan_entities[:5]:
                print(f"  - {node_id}: {name}")

        if orphan_relations:
            print(f"[WARNING] Found {len(orphan_relations)} orphan relations (first 5):")
            for node_id, content in orphan_relations[:5]:
                print(f"  - {node_id}: {content}...")

        # Check bipartite property (relations should only connect to entities)
        bipartite_violations = 0
        for edge in graph.edges():
            source_role = graph.nodes[edge[0]].get('role', 'unknown')
            target_role = graph.nodes[edge[1]].get('role', 'unknown')

            # Valid: relation -> entity or entity <- relation
            if not ((source_role == 'relation' and target_role == 'entity') or
                    (source_role == 'entity' and target_role == 'relation')):
                bipartite_violations += 1

        results['bipartite_violations'] = bipartite_violations
        print(f"[QUALITY] Bipartite violations: {bipartite_violations}")

        if bipartite_violations > 0:
            print("[ERROR] Graph violates bipartite property!")
        else:
            print("[OK] Graph maintains bipartite property")

        self.test_results['graph_structure'] = results
        return results

    async def test_chunk_indexing(self) -> Dict:
        """Test 2: Chunk indexing quality"""
        print("\n" + "="*80)
        print("TEST 2: CHUNK INDEXING QUALITY")
        print("="*80)

        results = {}

        # Load chunks from KV storage
        chunks_kv = self.rag.text_chunks
        all_chunks = await chunks_kv.get_by_ids([])  # Get all chunks

        results['total_chunks'] = len(all_chunks)
        print(f"[STATS] Total chunks in KV storage: {len(all_chunks)}")

        # Check chunk ID format (should be hash-based)
        sequential_ids = []
        hash_based_ids = []

        for chunk_id in all_chunks.keys():
            if chunk_id.startswith('chunk_'):
                sequential_ids.append(chunk_id)
            elif chunk_id.startswith('chunk-'):
                hash_based_ids.append(chunk_id)
            else:
                print(f"[WARNING] Unknown chunk ID format: {chunk_id}")

        results['sequential_chunk_ids'] = len(sequential_ids)
        results['hash_based_chunk_ids'] = len(hash_based_ids)

        print(f"[QUALITY] Hash-based chunk IDs: {len(hash_based_ids)}")
        print(f"[QUALITY] Sequential chunk IDs: {len(sequential_ids)}")

        if sequential_ids:
            print(f"[ERROR] Found {len(sequential_ids)} chunks with sequential IDs!")
            print(f"[ERROR] First 5 sequential IDs: {sequential_ids[:5]}")
        else:
            print("[OK] All chunks use hash-based IDs")

        # Check if chunks are indexed in vector DB
        if self.rag.vdb_chunks is not None:
            # NanoVectorDB doesn't have get_by_ids, count entries directly
            try:
                vdb_data = self.rag.vdb_chunks._NanoVectorDBStorage__client._data
                vdb_chunk_count = len(vdb_data) if vdb_data else 0
            except:
                vdb_chunk_count = 0

            results['vdb_chunk_count'] = vdb_chunk_count
            print(f"[STATS] Chunks in vector DB: {vdb_chunk_count}")

            if vdb_chunk_count == 0:
                print("[ERROR] No chunks indexed in vector DB (Path C will fail)!")
            elif vdb_chunk_count < len(all_chunks):
                print(f"[WARNING] Missing {len(all_chunks) - vdb_chunk_count} chunks in vector DB")
            else:
                print("[OK] All chunks indexed in vector DB")
        else:
            print("[WARNING] Vector DB for chunks not initialized")
            results['vdb_chunk_count'] = 0

        # Sample chunk content for inspection
        if all_chunks:
            sample_chunk_id = list(all_chunks.keys())[0]
            sample_chunk = all_chunks[sample_chunk_id]

            print(f"\n[SAMPLE] Sample chunk ID: {sample_chunk_id}")
            print(f"[SAMPLE] Sample chunk content (first 200 chars):")
            print(f"  {sample_chunk.get('content', 'NO_CONTENT')[:200]}...")
            print(f"[SAMPLE] Has doc_title: {bool(sample_chunk.get('doc_title'))}")
            print(f"[SAMPLE] Has doc_metadata: {bool(sample_chunk.get('doc_metadata'))}")

        self.test_results['chunk_indexing'] = results
        return results

    async def test_entity_extraction(self) -> Dict:
        """Test 3: Entity extraction quality"""
        print("\n" + "="*80)
        print("TEST 3: ENTITY EXTRACTION QUALITY")
        print("="*80)

        results = {}

        graph_storage = self.rag.chunk_entity_relation_graph
        graph = graph_storage._graph

        # Collect all entities
        entities = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('role') == 'entity':
                entities.append({
                    'id': node_id,
                    'name': node_data.get('entity_name', 'NO_NAME'),
                    'type': node_data.get('entity_type', 'UNKNOWN'),
                    'description': node_data.get('description', ''),
                    'weight': node_data.get('weight', 0.0),
                    'source_id': node_data.get('source_id', 'unknown'),
                })

        results['total_entities'] = len(entities)
        print(f"[STATS] Total entities extracted: {len(entities)}")

        # Check entity ID format
        sequential_entity_ids = []
        hash_based_entity_ids = []

        for entity in entities:
            if 'chunk_' in entity['source_id']:
                sequential_entity_ids.append(entity)
            elif 'chunk-' in entity['source_id']:
                hash_based_entity_ids.append(entity)

        results['entities_with_sequential_chunk_ids'] = len(sequential_entity_ids)
        results['entities_with_hash_chunk_ids'] = len(hash_based_entity_ids)

        print(f"[QUALITY] Entities with hash-based chunk IDs: {len(hash_based_entity_ids)}")
        print(f"[QUALITY] Entities with sequential chunk IDs: {len(sequential_entity_ids)}")

        if sequential_entity_ids:
            print(f"[ERROR] Found {len(sequential_entity_ids)} entities with sequential chunk IDs!")
            print(f"[ERROR] First 5 problematic entities:")
            for entity in sequential_entity_ids[:5]:
                print(f"  - {entity['name']}: source_id={entity['source_id']}")
        else:
            print("[OK] All entities use hash-based chunk IDs")

        # Entity type distribution
        type_dist = Counter(e['type'] for e in entities)
        results['entity_type_distribution'] = dict(type_dist)

        print(f"\n[STATS] Entity type distribution:")
        for etype, count in type_dist.most_common(10):
            print(f"  - {etype}: {count}")

        # Weight distribution
        weights = [e['weight'] for e in entities]
        if weights:
            avg_weight = sum(weights) / len(weights)
            max_weight = max(weights)
            min_weight = min(weights)

            results['avg_entity_weight'] = avg_weight
            results['max_entity_weight'] = max_weight
            results['min_entity_weight'] = min_weight

            print(f"\n[STATS] Entity weight distribution:")
            print(f"  - Average: {avg_weight:.2f}")
            print(f"  - Maximum: {max_weight:.2f}")
            print(f"  - Minimum: {min_weight:.2f}")

        # Check for multi-source entities (should have <SEP> in source_id)
        multi_source_entities = [e for e in entities if '<SEP>' in e['source_id']]
        results['multi_source_entities'] = len(multi_source_entities)

        print(f"\n[QUALITY] Multi-source entities (mentioned in multiple chunks): {len(multi_source_entities)}")

        if multi_source_entities:
            print(f"[OK] Provenance tracking working (first 5 multi-source entities):")
            for entity in multi_source_entities[:5]:
                source_count = entity['source_id'].count('<SEP>') + 1
                print(f"  - {entity['name']}: {source_count} source chunks")

        # Sample high-weight entities (most important)
        top_entities = sorted(entities, key=lambda e: e['weight'], reverse=True)[:10]

        print(f"\n[SAMPLE] Top 10 entities by weight (most important):")
        for i, entity in enumerate(top_entities, 1):
            print(f"  {i}. {entity['name']} (type={entity['type']}, weight={entity['weight']:.1f})")

        self.test_results['entity_extraction'] = results
        return results

    async def test_relation_extraction(self) -> Dict:
        """Test 4: Relation extraction quality"""
        print("\n" + "="*80)
        print("TEST 4: RELATION EXTRACTION QUALITY")
        print("="*80)

        results = {}

        graph_storage = self.rag.chunk_entity_relation_graph
        graph = graph_storage._graph

        # Collect all relations
        relations = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('role') == 'relation':
                relations.append({
                    'id': node_id,
                    'content': node_data.get('content', 'NO_CONTENT'),
                    'weight': node_data.get('weight', 0.0),
                    'source_id': node_data.get('source_id', 'unknown'),
                    'degree': graph.degree(node_id),  # How many entities connected
                })

        results['total_relations'] = len(relations)
        print(f"[STATS] Total relations extracted: {len(relations)}")

        # Check relation ID format (should start with 'rel-')
        correct_prefix = sum(1 for r in relations if r['id'].startswith('rel-'))
        results['relations_with_correct_prefix'] = correct_prefix

        print(f"[QUALITY] Relations with 'rel-' prefix: {correct_prefix}/{len(relations)}")

        if correct_prefix < len(relations):
            print(f"[ERROR] Found {len(relations) - correct_prefix} relations with incorrect prefix!")
        else:
            print("[OK] All relations use correct 'rel-' prefix")

        # Check connectivity (relations should connect to entities)
        connected_relations = [r for r in relations if r['degree'] > 0]
        orphan_relations = [r for r in relations if r['degree'] == 0]

        results['connected_relations'] = len(connected_relations)
        results['orphan_relations'] = len(orphan_relations)

        print(f"[QUALITY] Connected relations: {len(connected_relations)}")
        print(f"[QUALITY] Orphan relations (no entity links): {len(orphan_relations)}")

        if orphan_relations:
            print(f"[WARNING] Found {len(orphan_relations)} orphan relations (first 5):")
            for relation in orphan_relations[:5]:
                print(f"  - {relation['content'][:80]}...")

        # Average entities per relation
        if connected_relations:
            avg_entities = sum(r['degree'] for r in connected_relations) / len(connected_relations)
            results['avg_entities_per_relation'] = avg_entities
            print(f"[STATS] Average entities per relation: {avg_entities:.2f}")

        # Sample high-degree relations (connect to many entities)
        top_relations = sorted(relations, key=lambda r: r['degree'], reverse=True)[:5]

        print(f"\n[SAMPLE] Top 5 relations by connectivity:")
        for i, relation in enumerate(top_relations, 1):
            print(f"  {i}. Connects to {relation['degree']} entities")
            print(f"     Content: {relation['content'][:100]}...")

        self.test_results['relation_extraction'] = results
        return results

    async def test_three_path_retrieval(self) -> Dict:
        """Test 5: Three-path retrieval accuracy"""
        print("\n" + "="*80)
        print("TEST 5: THREE-PATH RETRIEVAL ACCURACY")
        print("="*80)

        results = {}

        # Test queries based on KUET admission document
        test_queries = [
            {
                'query': 'How many seats are there in CSE department?',
                'expected_info': ['120', 'CSE', 'Computer Science'],
            },
            {
                'query': 'What are the admission requirements for KUET?',
                'expected_info': ['GPA', '4.00', 'SSC', 'HSC'],
            },
            {
                'query': 'When is the admission exam?',
                'expected_info': ['January', '2025', '11', 'exam'],
            },
            {
                'query': 'What is the admission fee?',
                'expected_info': ['1100', '1200', 'fee', 'taka'],
            },
            {
                'query': 'How many departments are in KUET?',
                'expected_info': ['16', 'department', 'faculty'],
            },
        ]

        query_results = []

        for test_case in test_queries:
            query = test_case['query']
            expected = test_case['expected_info']

            print(f"\n[QUERY] Testing: {query}")

            # Test with hybrid mode (all three paths)
            param = QueryParam(
                mode='hybrid',
                top_k=10,
                enable_reranking=True,
            )

            try:
                contexts = await self.rag.aquery(query, param)

                # Convert contexts to string for inspection
                context_str = '\n'.join([
                    c.get('content', '') if isinstance(c, dict) else str(c)
                    for c in contexts
                ])

                # Check if expected information is present
                found_info = []
                missing_info = []

                for info in expected:
                    if info.lower() in context_str.lower():
                        found_info.append(info)
                    else:
                        missing_info.append(info)

                recall = len(found_info) / len(expected) if expected else 0

                query_results.append({
                    'query': query,
                    'contexts_returned': len(contexts),
                    'recall': recall,
                    'found_info': found_info,
                    'missing_info': missing_info,
                })

                print(f"[RESULT] Retrieved {len(contexts)} contexts")
                print(f"[RESULT] Recall: {recall:.2%} ({len(found_info)}/{len(expected)} expected info found)")

                if missing_info:
                    print(f"[WARNING] Missing expected info: {missing_info}")

                # Show first context snippet
                if contexts:
                    first_context = contexts[0]
                    content = first_context.get('content', str(first_context)) if isinstance(first_context, dict) else str(first_context)
                    print(f"[SAMPLE] First context (100 chars): {content[:100]}...")

            except Exception as e:
                print(f"[ERROR] Query failed: {e}")
                query_results.append({
                    'query': query,
                    'error': str(e),
                })

        # Calculate overall metrics
        successful_queries = [r for r in query_results if 'recall' in r]
        if successful_queries:
            avg_recall = sum(r['recall'] for r in successful_queries) / len(successful_queries)
            results['average_recall'] = avg_recall
            results['successful_queries'] = len(successful_queries)
            results['failed_queries'] = len(query_results) - len(successful_queries)

            print(f"\n[SUMMARY] Overall Performance:")
            print(f"  - Average recall: {avg_recall:.2%}")
            print(f"  - Successful queries: {len(successful_queries)}/{len(query_results)}")

        results['query_results'] = query_results
        self.test_results['three_path_retrieval'] = results
        return results

    async def test_question_answering(self) -> Dict:
        """Test 6: End-to-end question answering (if LLM available)"""
        print("\n" + "="*80)
        print("TEST 6: QUESTION ANSWERING CAPABILITY")
        print("="*80)

        # Check if LLM is configured
        if not os.getenv('OPENAI_API_KEY'):
            print("[SKIP] OpenAI API key not found, skipping QA test")
            return {'skipped': True, 'reason': 'No API key'}

        # Note: Full QA testing would require enabling LLM
        # For now, we just verify retrieval quality
        print("[INFO] QA testing requires LLM to be enabled")
        print("[INFO] Current test focuses on retrieval quality")
        print("[INFO] See TEST 5 results for retrieval accuracy")

        return {'status': 'See TEST 5 for retrieval metrics'}

    def generate_expert_report(self) -> str:
        """Generate comprehensive expert evaluation report"""
        print("\n" + "="*80)
        print("EXPERT EVALUATION REPORT")
        print("="*80)

        report_lines = []

        # Overall status
        report_lines.append("\n## OVERALL GRAPH QUALITY ASSESSMENT\n")

        # Check for critical issues
        critical_issues = []
        warnings = []
        successes = []

        # Graph structure checks
        if 'graph_structure' in self.test_results:
            gs = self.test_results['graph_structure']

            if gs.get('unknown_nodes', 0) > 0:
                critical_issues.append(f"Found {gs['unknown_nodes']} nodes with unknown role")

            if gs.get('bipartite_violations', 0) > 0:
                critical_issues.append(f"Graph violates bipartite property ({gs['bipartite_violations']} violations)")
            else:
                successes.append("Graph maintains bipartite property")

            if gs.get('orphan_entities', 0) > 0:
                warnings.append(f"Found {gs['orphan_entities']} orphan entities")
            else:
                successes.append("No orphan entities")

            if gs.get('orphan_relations', 0) > 0:
                warnings.append(f"Found {gs['orphan_relations']} orphan relations")
            else:
                successes.append("No orphan relations")

        # Chunk indexing checks
        if 'chunk_indexing' in self.test_results:
            ci = self.test_results['chunk_indexing']

            if ci.get('sequential_chunk_ids', 0) > 0:
                critical_issues.append(f"Found {ci['sequential_chunk_ids']} chunks with sequential IDs (should be hash-based)")
            else:
                successes.append("All chunks use hash-based IDs")

            if ci.get('vdb_chunk_count', 0) == 0:
                critical_issues.append("No chunks indexed in vector DB (Path C retrieval will fail)")
            else:
                successes.append(f"{ci['vdb_chunk_count']} chunks indexed in vector DB")

        # Entity extraction checks
        if 'entity_extraction' in self.test_results:
            ee = self.test_results['entity_extraction']

            if ee.get('entities_with_sequential_chunk_ids', 0) > 0:
                critical_issues.append(f"Found {ee['entities_with_sequential_chunk_ids']} entities with sequential chunk IDs")
            else:
                successes.append("All entities reference hash-based chunk IDs")

            if ee.get('multi_source_entities', 0) > 0:
                successes.append(f"{ee['multi_source_entities']} multi-source entities tracked (provenance working)")

        # Retrieval checks
        if 'three_path_retrieval' in self.test_results:
            tr = self.test_results['three_path_retrieval']

            avg_recall = tr.get('average_recall', 0)
            if avg_recall < 0.5:
                critical_issues.append(f"Low retrieval recall: {avg_recall:.2%}")
            elif avg_recall < 0.7:
                warnings.append(f"Moderate retrieval recall: {avg_recall:.2%}")
            else:
                successes.append(f"Good retrieval recall: {avg_recall:.2%}")

        # Print issues
        if critical_issues:
            report_lines.append("### CRITICAL ISSUES (Must Fix):\n")
            for issue in critical_issues:
                report_lines.append(f"- [CRITICAL] {issue}\n")

        if warnings:
            report_lines.append("\n### WARNINGS (Should Fix):\n")
            for warning in warnings:
                report_lines.append(f"- [WARNING] {warning}\n")

        if successes:
            report_lines.append("\n### SUCCESSES:\n")
            for success in successes:
                report_lines.append(f"- [OK] {success}\n")

        # Overall score
        total_checks = len(critical_issues) + len(warnings) + len(successes)
        if total_checks > 0:
            score = (len(successes) * 10 + len(warnings) * 5) / (total_checks * 10) * 100
            report_lines.append(f"\n### OVERALL QUALITY SCORE: {score:.1f}/100\n")

            if score >= 90:
                report_lines.append("**Status: EXCELLENT** - Graph is production-ready\n")
            elif score >= 70:
                report_lines.append("**Status: GOOD** - Minor improvements recommended\n")
            elif score >= 50:
                report_lines.append("**Status: FAIR** - Significant improvements needed\n")
            else:
                report_lines.append("**Status: POOR** - Major issues require fixing\n")

        # Recommendations
        report_lines.append("\n## RECOMMENDATIONS FOR IMPROVEMENT\n")

        if critical_issues:
            report_lines.append("\n### Immediate Actions Required:\n")
            if any('sequential' in issue.lower() for issue in critical_issues):
                report_lines.append("1. Fix chunk ID remapping issue - rebuild graph with fixed code\n")
            if any('vector db' in issue.lower() for issue in critical_issues):
                report_lines.append("2. Fix chunk vector DB indexing - ensure vdb_chunks is populated\n")

        if warnings:
            report_lines.append("\n### Suggested Improvements:\n")
            if any('orphan' in warning.lower() for warning in warnings):
                report_lines.append("1. Investigate orphan nodes - may indicate extraction issues\n")

        # Can we generate accurate answers?
        report_lines.append("\n## CAN BiG-RAG GENERATE ACCURATE ANSWERS?\n")

        if critical_issues:
            report_lines.append("**Answer: NO** - Critical issues prevent accurate retrieval\n")
            report_lines.append("Fix critical issues first, then rebuild and retest.\n")
        elif warnings and ('three_path_retrieval' in self.test_results and
                          self.test_results['three_path_retrieval'].get('average_recall', 0) < 0.7):
            report_lines.append("**Answer: PARTIALLY** - Retrieval works but accuracy is moderate\n")
            report_lines.append("Graph can generate answers but may miss some information.\n")
        else:
            report_lines.append("**Answer: YES** - Graph quality is sufficient for accurate answers\n")
            report_lines.append("BiG-RAG should generate accurate answers from this graph.\n")

        report = ''.join(report_lines)
        print(report)

        return report

    async def run_all_tests(self):
        """Run all quality tests"""
        print("\n" + "="*80)
        print(f"KUET UNIFIED GRAPH QUALITY TEST SUITE")
        print(f"Dataset: {self.dataset_name}")
        print(f"Path: {self.expr_path}")
        print("="*80)

        try:
            # Initialize
            await self.initialize()

            # Run tests
            await self.test_graph_structure()
            await self.test_chunk_indexing()
            await self.test_entity_extraction()
            await self.test_relation_extraction()
            await self.test_three_path_retrieval()
            await self.test_question_answering()

            # Generate report
            report = self.generate_expert_report()

            # Save results
            output_file = Path(f"test_scripts/kuet_unified_quality_report.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)

            print(f"\n[SAVED] Detailed results saved to: {output_file}")

            # Save text report
            report_file = Path(f"test_scripts/kuet_unified_quality_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"[SAVED] Text report saved to: {report_file}")

            print("\n[COMPLETE] All tests completed successfully!")

        except Exception as e:
            print(f"\n[ERROR] Test suite failed: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main entry point"""
    tester = KUETGraphQualityTester(dataset_name="kuet_unified")
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
