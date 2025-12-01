"""
Test Script: Diagnose Issue #1 (Zero LLM-Converted Relations) and Issue #2 (Counter Bug)

This script performs targeted checks to identify root causes:
1. Check if extractors populate linked_entities correctly
2. Check if validation preserves linked_entities metadata
3. Check which CASE each relation takes in Step 6.5
4. Verify counter accuracy

Run this AFTER rebuilding kuet_test with enhanced logging.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

def load_kv_store(dataset_path: str, store_name: str) -> dict:
    """Load KV store JSON file"""
    store_path = Path(dataset_path) / f"{store_name}.json"
    if not store_path.exists():
        print(f"[FAIL] KV store not found: {store_path}")
        return {}

    with open(store_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_graphml(dataset_path: str):
    """Analyze GraphML file for relation types"""
    graphml_path = Path(dataset_path) / "graph_chunk_entity_relation.graphml"
    if not graphml_path.exists():
        print(f"[FAIL] GraphML not found: {graphml_path}")
        return

    tree = ET.parse(graphml_path)
    root = tree.getroot()
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

    # Count relation types
    relation_nodes = []
    for node in root.findall('.//g:node', ns):
        node_id = node.get('id')
        if not node_id or not node_id.startswith('rel-'):
            continue

        role = node.find('.//g:data[@key="d0"]', ns)
        content = node.find('.//g:data[@key="d1"]', ns)

        if role is not None and role.text == 'relation':
            relation_nodes.append({
                'id': node_id,
                'content': content.text if content is not None else 'MISSING'
            })

    # Classify relations
    synthetic_count = 0
    table_count = 0
    paragraph_count = 0

    for rel in relation_nodes:
        if 'is mentioned as' in rel['content']:
            synthetic_count += 1
        elif any(x in rel['content'] for x in ['কোড', 'আসন', 'বিভাগ', 'code', 'seats', 'department']):
            table_count += 1
        else:
            paragraph_count += 1

    print("\n[GraphML Analysis]")
    print(f"Total relations: {len(relation_nodes)}")
    print(f"  - Synthetic ('is mentioned as'): {synthetic_count} ({synthetic_count/len(relation_nodes)*100:.1f}%)")
    print(f"  - Table-like (structured): {table_count} ({table_count/len(relation_nodes)*100:.1f}%)")
    print(f"  - Paragraph-like (other): {paragraph_count} ({paragraph_count/len(relation_nodes)*100:.1f}%)")

def parse_build_log(log_path: str):
    """Parse script_build.py output log to extract diagnostic info"""
    if not Path(log_path).exists():
        print(f"\n[FAIL] Log file not found: {log_path}")
        print("Please run: python script_build.py --data_source kuet_test --use_production_pipeline > build_diagnosis.log 2>&1")
        return

    print(f"\n[Parsing Build Log: {log_path}]")

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract diagnostic lines
    diagnostic_lines = [l for l in lines if '[DIAGNOSTIC]' in l]

    if not diagnostic_lines:
        print("[FAIL] No diagnostic output found in log!")
        print("Ensure bigrag.py has updated logging code.")
        return

    print(f"\nFound {len(diagnostic_lines)} diagnostic log entries:")
    print("-" * 80)

    # Group by section
    current_section = None
    for line in diagnostic_lines:
        if 'Sample relation after extraction' in line:
            current_section = "AFTER EXTRACTION"
            print(f"\n### {current_section} ###")
        elif 'Sample relation after validation' in line:
            current_section = "AFTER VALIDATION"
            print(f"\n### {current_section} ###")
        elif 'Processing relation' in line:
            current_section = "STEP 6.5 PROCESSING"
            print(f"\n### {current_section} ###")
        elif 'CASE distribution' in line:
            current_section = "FINAL COUNTS"
            print(f"\n### {current_section} ###")

        print(line.rstrip())

    print("-" * 80)

    # Extract key findings
    print("\n[Key Findings]")

    # Check if linked_entities preserved after validation
    after_extraction = [l for l in diagnostic_lines if 'after extraction' in l and 'linked_entities' in l]
    after_validation = [l for l in diagnostic_lines if 'after validation' in l and 'linked_entities' in l]

    if after_extraction and after_validation:
        print("\n[Issue #1 Analysis: linked_entities Preservation]")
        print("After extraction:", after_extraction[0].strip())
        print("After validation:", after_validation[0].strip())

        if 'EMPTY' in after_extraction[0]:
            print("  → [FINDING] Extractor DID NOT populate linked_entities")
        else:
            print("  → [FINDING] Extractor DID populate linked_entities")

        if 'EMPTY' in after_validation[0] and 'EMPTY' not in after_extraction[0]:
            print("  → [ROOT CAUSE FOUND] Validation CLEARED linked_entities!")
        elif 'EMPTY' in after_validation[0]:
            print("  → [FINDING] linked_entities still empty after validation")
        else:
            print("  → [FINDING] Validation preserved linked_entities")

    # Check CASE distribution
    case_dist_lines = [l for l in diagnostic_lines if 'CASE distribution' in l]
    if case_dist_lines:
        print("\n[Issue #1 Analysis: CASE Distribution]")
        print(case_dist_lines[0].strip())

        # Parse counts
        import re
        match = re.search(r'CASE1=(\d+), CASE2=(\d+), CASE3=(\d+), Total=(\d+)', case_dist_lines[0])
        if match:
            case1, case2, case3, total = map(int, match.groups())
            print(f"  → CASE 1 (LLM names): {case1} relations ({case1/total*100:.1f}%)")
            print(f"  → CASE 2 (entity IDs): {case2} relations ({case2/total*100:.1f}%)")
            print(f"  → CASE 3 (fallback): {case3} relations ({case3/total*100:.1f}%)")

            if case1 + case2 + case3 != total:
                print(f"  → [BUG DETECTED] Sum mismatch: {case1}+{case2}+{case3}={case1+case2+case3} != {total}")
            else:
                print(f"  → [OK] Sum matches total: {case1+case2+case3} == {total}")

    # Check counter verification
    counter_lines = [l for l in diagnostic_lines if 'Counter verification' in l]
    if counter_lines:
        print("\n[Issue #2 Analysis: Counter Bug]")
        print(counter_lines[0].strip())

        # Compare with CASE distribution
        if case_dist_lines:
            match_counter = re.search(r'relations_llm_linked=(\d+), relations_id_remapped=(\d+), relations_relinked=(\d+)', counter_lines[0])
            if match_counter and match:
                llm_linked, id_remapped, relinked = map(int, match_counter.groups())
                case1, case2, case3, total = map(int, match.groups())

                print(f"\nCounter vs CASE comparison:")
                print(f"  relations_llm_linked ({llm_linked}) == CASE1 ({case1})? {llm_linked == case1}")
                print(f"  relations_relinked ({relinked}) == CASE3 ({case3})? {relinked == case3}")

                if llm_linked != case1:
                    print(f"  → [BUG DETECTED] Counter mismatch for LLM-linked!")
                if relinked != case3:
                    print(f"  → [BUG DETECTED] Counter mismatch for fallback-linked!")

                if llm_linked == case1 and relinked == case3:
                    print(f"  → [OK] Counters match CASE distribution")

def main():
    dataset_path = "d:/BiG-RAG/expr/kuet_test"
    log_path = "build_diagnosis.log"

    print("=" * 80)
    print("BiG-RAG Issue #1 & #2 Diagnostic Tool")
    print("=" * 80)

    # Step 1: Analyze GraphML
    print("\n[STEP 1] Analyzing GraphML file...")
    analyze_graphml(dataset_path)

    # Step 2: Parse build log
    print("\n[STEP 2] Parsing build log for diagnostic output...")
    parse_build_log(log_path)

    print("\n" + "=" * 80)
    print("[DIAGNOSTIC COMPLETE]")
    print("\nNext steps:")
    print("1. If log file not found, run:")
    print("   python script_build.py --data_source kuet_test --use_production_pipeline > build_diagnosis.log 2>&1")
    print("2. Re-run this script to analyze the log")
    print("3. Share build_diagnosis.log with the analysis results")
    print("=" * 80)

if __name__ == "__main__":
    main()
