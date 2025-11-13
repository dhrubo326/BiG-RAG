"""
BiG-RAG Knowledge Graph Orphan Node Detection Test

Analyzes a GraphML file to detect orphan nodes (nodes with no edges).
Provides before/after comparison for improvement tracking.

Usage:
    python test_orphan_detection.py football
    python test_orphan_detection.py football --compare expr/football_backup/graph_chunk_entity_relation.graphml
"""

import asyncio
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
from typing import Dict, List, Tuple


def parse_graphml_structure(graph_path: Path) -> Dict:
    """
    Parse GraphML file and extract node/edge statistics.

    Returns:
        dict: {
            "entity_nodes": list[str],
            "relation_nodes": list[str],
            "edges": list[tuple],
            "orphan_entities": list[str],
            "orphan_relations": list[str],
            "orphan_relation_details": list[dict]
        }
    """
    tree = ET.parse(graph_path)
    root = tree.getroot()
    ns = {'gml': 'http://graphml.graphdrawing.org/xmlns'}

    # Collect nodes by type
    entity_nodes = []
    relation_nodes = []
    relation_node_details = {}  # id -> {content, weight}

    for node in root.findall('.//gml:node', ns):
        node_id = node.get('id')
        role = None
        content = None
        weight = None

        for data in node.findall('gml:data', ns):
            key = data.get('key')
            if key == 'd0':  # role
                role = data.text
            elif key == 'd1':  # content (for relations)
                content = data.text
            elif key == 'd2':  # weight
                try:
                    weight = float(data.text)
                except (ValueError, TypeError):
                    weight = 0.0

        if role == 'entity':
            entity_nodes.append(node_id)
        elif role == 'relation':
            relation_nodes.append(node_id)
            relation_node_details[node_id] = {
                'content': content or '',
                'weight': weight or 0.0
            }

    # Collect edges
    edges = []
    connected_nodes = set()
    for edge in root.findall('.//gml:edge', ns):
        source = edge.get('source')
        target = edge.get('target')
        edges.append((source, target))
        connected_nodes.add(source)
        connected_nodes.add(target)

    # Find orphans
    all_nodes = set(entity_nodes + relation_nodes)
    orphan_nodes = all_nodes - connected_nodes

    orphan_entities = [n for n in orphan_nodes if n in entity_nodes]
    orphan_relations = [n for n in orphan_nodes if n in relation_nodes]

    # Collect orphan relation details
    orphan_relation_details = []
    for orphan_id in orphan_relations:
        if orphan_id in relation_node_details:
            orphan_relation_details.append({
                'id': orphan_id,
                'content': relation_node_details[orphan_id]['content'],
                'weight': relation_node_details[orphan_id]['weight']
            })

    return {
        'entity_nodes': entity_nodes,
        'relation_nodes': relation_nodes,
        'edges': edges,
        'orphan_entities': orphan_entities,
        'orphan_relations': orphan_relations,
        'orphan_relation_details': orphan_relation_details,
        'all_nodes': len(all_nodes)
    }


def print_orphan_report(stats: Dict, dataset_name: str, label: str = "CURRENT"):
    """Print formatted orphan analysis report"""

    total_nodes = stats['all_nodes']
    entity_count = len(stats['entity_nodes'])
    relation_count = len(stats['relation_nodes'])
    edge_count = len(stats['edges'])

    orphan_count = len(stats['orphan_entities']) + len(stats['orphan_relations'])
    orphan_entity_count = len(stats['orphan_entities'])
    orphan_relation_count = len(stats['orphan_relations'])

    orphan_rate = orphan_count / total_nodes if total_nodes > 0 else 0
    orphan_relation_rate = orphan_relation_count / relation_count if relation_count > 0 else 0

    avg_degree = (edge_count * 2) / total_nodes if total_nodes > 0 else 0
    avg_edges_per_relation = edge_count / relation_count if relation_count > 0 else 0

    print(f"\n{'='*80}")
    print(f"ORPHAN NODE ANALYSIS: {dataset_name} ({label})")
    print(f"{'='*80}")
    print(f"\nGraph Statistics:")
    print(f"   Total Nodes:        {total_nodes}")
    print(f"   Entity Nodes:       {entity_count}")
    print(f"   Relation Nodes:     {relation_count}")
    print(f"   Total Edges:        {edge_count}")
    print(f"   Avg Degree:         {avg_degree:.2f}")
    print(f"   Avg Edges/Relation: {avg_edges_per_relation:.2f}")

    print(f"\nOrphan Analysis:")
    print(f"   Total Orphans:      {orphan_count} ({orphan_rate:.1%})")
    print(f"   Orphan Entities:    {orphan_entity_count}")
    print(f"   Orphan Relations:   {orphan_relation_count} ({orphan_relation_rate:.1%})")

    # Quality assessment
    print(f"\nQuality Assessment:")
    if orphan_relation_rate < 0.05:
        status = "[OK] EXCELLENT (Production Ready)"
    elif orphan_relation_rate < 0.10:
        status = "[WARNING] GOOD (Minor Issues)"
    elif orphan_relation_rate < 0.20:
        status = "[WARNING] NEEDS IMPROVEMENT"
    else:
        status = "[CRITICAL] CRITICAL ISSUES"

    print(f"   Orphan Relation Rate: {orphan_relation_rate:.1%}")
    print(f"   Status: {status}")
    print(f"   Target: <5% for production quality")

    # Show sample orphan relations
    if stats['orphan_relation_details']:
        print(f"\n{'='*80}")
        print(f"SAMPLE ORPHAN RELATIONS (First 5):")
        print(f"{'='*80}")

        for i, detail in enumerate(stats['orphan_relation_details'][:5], 1):
            content = detail['content']
            weight = detail['weight']
            # Truncate long content
            display_content = content[:100] + "..." if len(content) > 100 else content
            print(f"\n{i}. ID: {detail['id']}")
            print(f"   Weight: {weight}")
            print(f"   Content: {display_content}")


def print_comparison_report(before_stats: Dict, after_stats: Dict, dataset_name: str):
    """Print before/after comparison report"""

    before_orphan_rate = (
        len(before_stats['orphan_relations']) / len(before_stats['relation_nodes'])
        if before_stats['relation_nodes'] else 0
    )
    after_orphan_rate = (
        len(after_stats['orphan_relations']) / len(after_stats['relation_nodes'])
        if after_stats['relation_nodes'] else 0
    )

    improvement = before_orphan_rate - after_orphan_rate
    improvement_pct = (improvement / before_orphan_rate * 100) if before_orphan_rate > 0 else 0

    before_avg_edges = (
        len(before_stats['edges']) / len(before_stats['relation_nodes'])
        if before_stats['relation_nodes'] else 0
    )
    after_avg_edges = (
        len(after_stats['edges']) / len(after_stats['relation_nodes'])
        if after_stats['relation_nodes'] else 0
    )

    print(f"\n{'='*80}")
    print(f"BEFORE/AFTER COMPARISON: {dataset_name}")
    print(f"{'='*80}")

    print(f"\nOrphan Relation Rate:")
    print(f"   Before: {before_orphan_rate:.1%} ({len(before_stats['orphan_relations'])}/{len(before_stats['relation_nodes'])})")
    print(f"   After:  {after_orphan_rate:.1%} ({len(after_stats['orphan_relations'])}/{len(after_stats['relation_nodes'])})")
    print(f"   Change: {improvement:+.1%} ({improvement_pct:+.0f}%)")

    print(f"\nGraph Connectivity:")
    print(f"   Edges Before: {len(before_stats['edges'])}")
    print(f"   Edges After:  {len(after_stats['edges'])}")
    print(f"   Change: {len(after_stats['edges']) - len(before_stats['edges']):+d}")

    print(f"\nAvg Edges per Relation:")
    print(f"   Before: {before_avg_edges:.2f}")
    print(f"   After:  {after_avg_edges:.2f}")
    print(f"   Change: {after_avg_edges - before_avg_edges:+.2f}")

    # Overall verdict
    print(f"\n{'='*80}")
    print(f"OVERALL VERDICT:")
    print(f"{'='*80}")

    if improvement > 0.15:
        verdict = "[OK] EXCELLENT IMPROVEMENT (>15% reduction)"
    elif improvement > 0.10:
        verdict = "[OK] GOOD IMPROVEMENT (10-15% reduction)"
    elif improvement > 0.05:
        verdict = "[WARNING] MODERATE IMPROVEMENT (5-10% reduction)"
    elif improvement > 0:
        verdict = "[WARNING] MINOR IMPROVEMENT (<5% reduction)"
    else:
        verdict = "[FAIL] NO IMPROVEMENT or REGRESSION"

    print(f"   {verdict}")

    if after_orphan_rate < 0.05:
        print(f"   [OK] Target achieved: Orphan rate now <5%")
    else:
        print(f"   [WARNING] Target not yet achieved: Orphan rate still {after_orphan_rate:.1%} (target: <5%)")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python test_orphan_detection.py <dataset_name> [--compare <before_graphml_path>]")
        print("Example: python test_orphan_detection.py football")
        print("Example: python test_orphan_detection.py football --compare expr/football_backup/graph_chunk_entity_relation.graphml")
        sys.exit(1)

    dataset_name = sys.argv[1]
    current_graph_path = Path(f"expr/{dataset_name}/graph_chunk_entity_relation.graphml")

    if not current_graph_path.exists():
        print(f"ERROR: Graph not found at {current_graph_path}")
        sys.exit(1)

    # Parse current graph
    print(f"Parsing current graph: {current_graph_path}")
    current_stats = parse_graphml_structure(current_graph_path)

    # Check for comparison mode
    if len(sys.argv) >= 4 and sys.argv[2] == '--compare':
        before_graph_path = Path(sys.argv[3])
        if not before_graph_path.exists():
            print(f"ERROR: Before graph not found at {before_graph_path}")
            sys.exit(1)

        print(f"Parsing before graph: {before_graph_path}")
        before_stats = parse_graphml_structure(before_graph_path)

        # Print before report
        print_orphan_report(before_stats, dataset_name, label="BEFORE")

        # Print after report
        print_orphan_report(current_stats, dataset_name, label="AFTER")

        # Print comparison
        print_comparison_report(before_stats, current_stats, dataset_name)
    else:
        # Print single report
        print_orphan_report(current_stats, dataset_name, label="CURRENT")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
