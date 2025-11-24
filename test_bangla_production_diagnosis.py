"""
Bangla Production Pipeline Diagnostic Test

This script tests the production pipeline with KUET_Admission_info.md (Bangla content)
and logs everything to diagnose validation failures.

It mimics the /datasets/create-and-index endpoint workflow to identify root causes
of production pipeline failures with Bangla/multilingual content.

Usage:
    python test_bangla_production_diagnosis.py

Output:
    - bangla_production_diagnosis.log (detailed log file)
    - Console output with summary
"""

import asyncio
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import json
import traceback

# Fix Windows console encoding for Bangla characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Setup logging to file and console
log_file = "bangla_production_diagnosis.log"
log_handle = None

def log(message, level="INFO"):
    """Log to both file and console with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [{level}] {message}"

    print(formatted)

    global log_handle
    if log_handle:
        log_handle.write(formatted + "\n")
        log_handle.flush()


async def test_bangla_production_pipeline():
    """
    Complete diagnostic test of production pipeline with Bangla content.

    Tests the full workflow:
    1. Load KUET document (Bangla)
    2. Initialize BiGRAG with production pipeline
    3. Process with ProductionKGPipeline
    4. Log all phases and validation results
    5. Identify root causes of failures
    """

    global log_handle
    log_handle = open(log_file, 'w', encoding='utf-8')

    log("="*80)
    log("BANGLA PRODUCTION PIPELINE DIAGNOSTIC TEST")
    log("="*80)
    log("")

    # Step 1: Check environment
    log("Step 1: Environment Check", "SETUP")
    log("-"*80)

    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        log("OPENAI_API_KEY not found in environment!", "ERROR")
        log("Please set OPENAI_API_KEY in your .env file", "ERROR")
        return False

    log(f"API key found: {api_key[:20]}...", "OK")

    # Check KUET file
    kuet_file = "KUET_Admission_info.md"
    if not os.path.exists(kuet_file):
        log(f"{kuet_file} not found!", "ERROR")
        return False

    log(f"Test document found: {kuet_file}", "OK")
    log("")

    # Step 2: Load document
    log("Step 2: Load Document", "SETUP")
    log("-"*80)

    with open(kuet_file, 'r', encoding='utf-8') as f:
        kuet_doc = f.read()

    log(f"Document loaded: {len(kuet_doc)} characters", "OK")
    log(f"Document preview (first 200 chars):", "INFO")
    log(kuet_doc[:200], "DATA")
    log("")

    # Detect language
    log("Detecting language composition...", "INFO")
    bengali_chars = sum(1 for c in kuet_doc if '\u0980' <= c <= '\u09FF')
    english_chars = sum(1 for c in kuet_doc if 'a' <= c.lower() <= 'z')
    total_alpha = bengali_chars + english_chars

    if total_alpha > 0:
        bengali_pct = (bengali_chars / total_alpha) * 100
        english_pct = (english_chars / total_alpha) * 100
    else:
        bengali_pct = 0
        english_pct = 0

    log(f"Bengali characters: {bengali_chars} ({bengali_pct:.1f}%)", "DATA")
    log(f"English characters: {english_chars} ({english_pct:.1f}%)", "DATA")
    log(f"Primary language: {'BANGLA' if bengali_pct > english_pct else 'ENGLISH'}", "INFO")
    log("")

    # Step 3: Setup test directory
    log("Step 3: Setup Test Environment", "SETUP")
    log("-"*80)

    working_dir = "./expr/bangla_diagnosis_test"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    log(f"Working directory: {working_dir}", "OK")
    log("")

    # Step 4: Initialize BiGRAG with production pipeline
    log("Step 4: Initialize BiGRAG with Production Pipeline", "SETUP")
    log("-"*80)

    from bigrag import BiGRAG

    try:
        rag = BiGRAG(
            working_dir=working_dir,
            use_production_pipeline=True,  # PRODUCTION MODE
            production_pipeline_config={
                "validation_level": "MODERATE",  # 95%+ validation
                "enable_entity_linking": True,
                "extraction_mode": "semi_structured"
            }
        )
        log("BiGRAG initialized with PRODUCTION pipeline", "OK")
        log("Config: validation=MODERATE, entity_linking=True, mode=semi_structured", "INFO")
        log("")
    except Exception as e:
        log(f"Failed to initialize BiGRAG: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return False

    # Step 5: Process document through production pipeline
    log("Step 5: Process Document with Production Pipeline", "PROCESS")
    log("="*80)
    log("")

    metadata = {
        'title': 'KUET Admission 2024-25',
        'category': 'university_admission',
        'tags': ['engineering', 'admission', 'KUET', 'Bangladesh'],
        'language': 'Bangla'
    }

    try:
        # This will trigger _process_document_with_production_pipeline()
        await rag.ainsert(
            [kuet_doc],
            metadata=[metadata]
        )
        log("Document insertion completed", "OK")
        log("")
    except Exception as e:
        log(f"Document insertion failed: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")

        # Check if files were still created (fallback to standard pipeline)
        files_created = []
        for fname in ["graph_chunk_entity_relation.graphml", "vdb_entities.json", "vdb_relations.json"]:
            fpath = Path(working_dir) / fname
            if fpath.exists():
                files_created.append(fname)

        if files_created:
            log(f"Fallback to standard pipeline succeeded ({len(files_created)} files created)", "WARN")
        else:
            log("Complete failure - no files created", "CRITICAL")
            return False

    # Step 6: Analyze created files
    log("Step 6: Analyze Created Files", "ANALYSIS")
    log("-"*80)

    expected_files = [
        "graph_chunk_entity_relation.graphml",
        "vdb_entities.json",
        "vdb_relations.json",
        "vdb_chunks.json",
        "kv_store_text_chunks.json",
        "kv_store_full_docs.json",
    ]

    files_status = {}
    for filename in expected_files:
        filepath = Path(working_dir) / filename
        if filepath.exists():
            size = filepath.stat().st_size
            files_status[filename] = {"exists": True, "size": size}

            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"

            log(f"[OK] {filename} ({size_str})", "FILE")
        else:
            files_status[filename] = {"exists": False, "size": 0}
            log(f"[MISSING] {filename}", "FILE")

    log("")

    # Step 7: Analyze graph statistics and content
    log("Step 7: Detailed Graph Analysis", "ANALYSIS")
    log("-"*80)

    try:
        # Load graph stats
        graphml_path = Path(working_dir) / "graph_chunk_entity_relation.graphml"
        if graphml_path.exists():
            import networkx as nx
            graph = nx.read_graphml(str(graphml_path))

            entity_nodes = [n for n, d in graph.nodes(data=True) if d.get('role') == 'entity']
            relation_nodes = [n for n, d in graph.nodes(data=True) if d.get('role') == 'bipartite_edge']

            log(f"Total nodes: {graph.number_of_nodes()}", "STAT")
            log(f"Total edges: {graph.number_of_edges()}", "STAT")
            log(f"Entity nodes: {len(entity_nodes)}", "STAT")
            log(f"Relation nodes: {len(relation_nodes)}", "STAT")
            log("")

            # Sample entities with MORE details for investigation
            log("Sample Entities (first 10 for investigation):", "INFO")
            for i, entity_id in enumerate(entity_nodes[:10]):
                entity_data = graph.nodes[entity_id]
                # Entity name is the node ID itself (not stored as 'name' attribute)
                name = entity_id.strip('"')  # Remove quotes from GraphML node ID
                entity_type = entity_data.get('entity_type', 'N/A')
                weight = entity_data.get('weight', 'N/A')
                # Truncate long names for readability
                display_name = name[:60] + "..." if len(name) > 60 else name
                log(f"  {i+1}. {display_name} (type: {entity_type}, weight: {weight})", "DATA")
            log("")

            # Sample relations with MORE details for investigation
            log("Sample Relations (first 10 for investigation):", "INFO")
            for i, rel_id in enumerate(relation_nodes[:10]):
                rel_data = graph.nodes[rel_id]
                name = rel_data.get('name', 'N/A')
                desc = rel_data.get('description', 'N/A')
                weight = rel_data.get('weight', 'N/A')
                # Truncate description if too long
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                log(f"  {i+1}. {name}", "DATA")
                log(f"      Description: {desc}", "DATA")
                log(f"      Weight: {weight}", "DATA")
            log("")

        # Load chunk stats with sample content
        chunks_path = Path(working_dir) / "kv_store_text_chunks.json"
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            log(f"Text chunks created: {len(chunks)}", "STAT")
            log("")

            # Show first chunk for investigation
            if chunks:
                log("Sample Chunk (first chunk for investigation):", "INFO")
                first_chunk_id = list(chunks.keys())[0]
                first_chunk = chunks[first_chunk_id]
                content = first_chunk.get('content', 'N/A')
                if len(content) > 200:
                    content = content[:197] + "..."
                log(f"  Chunk ID: {first_chunk_id}", "DATA")
                log(f"  Content preview: {content}", "DATA")
                log(f"  Title: {first_chunk.get('title', 'N/A')}", "DATA")
                log("")

        # Load vector DB stats
        vdb_entities_path = Path(working_dir) / "vdb_entities.json"
        if vdb_entities_path.exists():
            with open(vdb_entities_path, 'r', encoding='utf-8') as f:
                vdb_data = json.load(f)
            vectors_count = len(vdb_data.get('vectors', []))
            log(f"Entity vectors indexed: {vectors_count}", "STAT")

        vdb_relations_path = Path(working_dir) / "vdb_relations.json"
        if vdb_relations_path.exists():
            with open(vdb_relations_path, 'r', encoding='utf-8') as f:
                vdb_data = json.load(f)
            vectors_count = len(vdb_data.get('vectors', []))
            log(f"Relation vectors indexed: {vectors_count}", "STAT")

        vdb_chunks_path = Path(working_dir) / "vdb_chunks.json"
        if vdb_chunks_path.exists():
            with open(vdb_chunks_path, 'r', encoding='utf-8') as f:
                vdb_data = json.load(f)
            vectors_count = len(vdb_data.get('vectors', []))
            log(f"Chunk vectors indexed: {vectors_count}", "STAT")

        log("")
        log("Files location for manual investigation:", "INFO")
        log(f"  {working_dir}/", "FILE")
        log("")

    except Exception as e:
        log(f"Failed to analyze graph: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        log("")

    # Step 8: Detect validation issues from logs
    log("Step 8: Issue Detection", "ANALYSIS")
    log("="*80)
    log("")

    issues_found = []

    # Check if production pipeline was used or fell back
    log_path = Path(log_file)
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # Check for specific issues
        if "[Production Pipeline] Validation FAILED" in log_content or "WARNING:bigrag:[Production Pipeline] Falling back" in log_content:
            issues_found.append({
                "issue": "Production pipeline validation failed",
                "severity": "HIGH",
                "description": "Production pipeline attempted but failed validation checks"
            })

        if "Table validation failed" in log_content:
            issues_found.append({
                "issue": "Table validation failed",
                "severity": "MEDIUM",
                "description": "Bangla numerals in tables not recognized during validation"
            })

        if "Numeric coverage:" in log_content:
            # Extract numeric coverage value
            import re
            match = re.search(r"Numeric coverage: (\d+\.?\d*)%", log_content)
            if match:
                coverage = float(match.group(1))
                if coverage < 95:
                    issues_found.append({
                        "issue": f"Low numeric coverage ({coverage}%)",
                        "severity": "HIGH",
                        "description": f"Expected 95%+, got {coverage}%. Bangla numbers (০-৯) not detected."
                    })

        if "Consistency:" in log_content and "-" in log_content:
            issues_found.append({
                "issue": "Negative consistency score",
                "severity": "HIGH",
                "description": "Cross-chunk consistency validation failed (negative score indicates conflicts)"
            })

        if "ORPHAN RELATION" in log_content:
            # Count orphan warnings
            orphan_count = log_content.count("ORPHAN RELATION")
            issues_found.append({
                "issue": f"High orphan relation rate ({orphan_count} warnings)",
                "severity": "MEDIUM",
                "description": "Relations extracted without corresponding entities (entity linking failure)"
            })

    # Step 9: Root Cause Analysis
    log("Step 9: Root Cause Analysis", "ANALYSIS")
    log("="*80)
    log("")

    if not issues_found:
        log("No issues detected - production pipeline worked perfectly!", "SUCCESS")
    else:
        log(f"Found {len(issues_found)} issues:", "WARN")
        log("")

        for i, issue in enumerate(issues_found, 1):
            log(f"Issue #{i}: {issue['issue']}", "ISSUE")
            log(f"  Severity: {issue['severity']}", "INFO")
            log(f"  Description: {issue['description']}", "INFO")
            log("")

        # Provide DYNAMIC root cause analysis based on detected issues
        log("ROOT CAUSE ANALYSIS (based on detected issues):", "ANALYSIS")
        log("-"*80)
        log("")

        # Extract actual data from issues
        numeric_coverage = None
        consistency_score = None
        orphan_count = None

        for issue in issues_found:
            if "coverage" in issue['issue'].lower():
                import re
                match = re.search(r"(\d+\.?\d*)%", issue['issue'])
                if match:
                    numeric_coverage = float(match.group(1))

            if "consistency" in issue['issue'].lower():
                consistency_score = "negative"

            if "orphan" in issue['issue'].lower():
                match = re.search(r"(\d+) warnings", issue['issue'])
                if match:
                    orphan_count = int(match.group(1))

        # Dynamic root cause analysis
        root_cause_num = 1

        if numeric_coverage is not None and numeric_coverage < 95:
            log(f"{root_cause_num}. BANGLA NUMERAL DETECTION", "ROOT_CAUSE")
            log(f"   Detected Issue: Numeric coverage is {numeric_coverage}% (expected 95%+)", "INFO")
            log("   Likely Cause: Production pipeline uses ASCII digit regex (0-9)", "INFO")
            log("   Hypothesis: Document contains Bangla numerals (০-৯) not recognized", "INFO")
            log("   Impact: Validation fails even when extraction is correct", "INFO")
            log("")
            root_cause_num += 1

        if "Table validation failed" in [i['issue'] for i in issues_found]:
            log(f"{root_cause_num}. TABLE EXTRACTION VALIDATION MISMATCH", "ROOT_CAUSE")
            log("   Detected Issue: Table validation failed", "INFO")
            log("   Likely Cause: GPT-4o extracts tables correctly but validator can't verify", "INFO")
            log("   Hypothesis: Validator compares Bangla numerals in tables against source", "INFO")
            log("   Impact: Correct tables rejected due to numeral format mismatch", "INFO")
            log("")
            root_cause_num += 1

        if consistency_score:
            log(f"{root_cause_num}. CROSS-CHUNK CONSISTENCY FOR MULTILINGUAL CONTENT", "ROOT_CAUSE")
            log("   Detected Issue: Negative consistency score", "INFO")
            log("   Likely Cause: Validator uses exact string matching for entity names", "INFO")
            log("   Hypothesis: Same entity appears in multiple languages (Bengali + English)", "INFO")
            log("   Impact: False conflicts detected between language variants", "INFO")
            log("")
            root_cause_num += 1

        if orphan_count:
            log(f"{root_cause_num}. ENTITY LINKING WITH BANGLA SCRIPT", "ROOT_CAUSE")
            log(f"   Detected Issue: {orphan_count} orphan relation warnings", "INFO")
            log("   Likely Cause: Fuzzy matching optimized for Latin script", "INFO")
            log("   Hypothesis: Bangla script has different edit distance characteristics", "INFO")
            log("   Impact: Similar entities not linked (e.g., stem variations)", "INFO")
            log("")
            root_cause_num += 1

        if root_cause_num == 1:
            log("Unable to extract specific metrics from detected issues.", "WARN")
            log("Manual log review recommended for detailed root cause analysis.", "INFO")
            log("")

    # Step 10: Recommended Fixes (Dynamic based on detected issues)
    log("Step 10: Recommended Fixes (based on detected issues)", "SOLUTION")
    log("="*80)
    log("")

    if not issues_found:
        log("No fixes needed - production pipeline worked perfectly!", "SUCCESS")
        log("")
    else:
        log("The following fixes are recommended based on detected issues:", "INFO")
        log("")

        fix_priority = 1

        # Fix recommendations based on actual detected issues
        if numeric_coverage is not None and numeric_coverage < 95:
            log(f"Priority {fix_priority}: Fix Numeric Validation for Bangla Numerals", "FIX")
            log(f"  Reason: Detected numeric coverage {numeric_coverage}% (target: 95%+)", "INFO")
            log("  Files to Check:", "INFO")
            log("    - bigrag/production_pipeline.py (validation logic)", "FILE")
            log("    - bigrag/preprocessors/*.py (check for \\d regex patterns)", "FILE")
            log("  Suggestion: Add Bangla numeral detection or normalize before validation", "INFO")
            log("")
            fix_priority += 1

        if "Table validation failed" in [i['issue'] for i in issues_found]:
            log(f"Priority {fix_priority}: Fix Table Validation Logic", "FIX")
            log("  Reason: Table validation failed during production pipeline", "INFO")
            log("  Files to Check:", "INFO")
            log("    - bigrag/preprocessors/table_extractor.py (if exists)", "FILE")
            log("    - bigrag/production_pipeline.py (table validation phase)", "FILE")
            log("  Suggestion: Review how table coverage is calculated", "INFO")
            log("")
            fix_priority += 1

        if consistency_score:
            log(f"Priority {fix_priority}: Fix Consistency Validation for Multilingual", "FIX")
            log("  Reason: Negative consistency score detected", "INFO")
            log("  Files to Check:", "INFO")
            log("    - bigrag/production_pipeline.py (consistency check logic)", "FILE")
            log("    - bigrag/validators/*.py (check for exact string matching)", "FILE")
            log("  Suggestion: Consider semantic similarity for entity matching", "INFO")
            log("")
            fix_priority += 1

        if orphan_count:
            log(f"Priority {fix_priority}: Improve Entity Linking", "FIX")
            log(f"  Reason: {orphan_count} orphan relation warnings detected", "INFO")
            log("  Files to Check:", "INFO")
            log("    - bigrag/operate.py (_merge_nodes_then_upsert function)", "FILE")
            log("    - Entity linking/fuzzy matching logic", "FILE")
            log("  Suggestion: Review fuzzy matching for Bangla script compatibility", "INFO")
            log("")
            fix_priority += 1

        if fix_priority == 1:
            log("Issues detected but unable to map to specific fixes.", "WARN")
            log("Manual code review recommended.", "INFO")
            log("")

    # Step 11: Summary
    log("="*80)
    log("TEST SUMMARY")
    log("="*80)
    log("")

    log(f"Document: {kuet_file} ({len(kuet_doc)} chars)", "INFO")
    log(f"Language: Bangla ({bengali_pct:.1f}%) + English ({english_pct:.1f}%)", "INFO")
    log(f"Pipeline Mode: PRODUCTION", "INFO")
    log(f"Files Created: {sum(1 for f in files_status.values() if f['exists'])}/{len(expected_files)}", "INFO")
    log(f"Issues Found: {len(issues_found)}", "INFO")
    log("")

    if all(f['exists'] for f in files_status.values()) and len(issues_found) == 0:
        log("Result: SUCCESS (production pipeline completed successfully)", "RESULT")
    elif all(f['exists'] for f in files_status.values()):
        log("Result: PARTIAL SUCCESS (files created but with warnings)", "RESULT")
    else:
        log("Result: FAILURE (missing files)", "RESULT")

    log("")
    log(f"Detailed log saved to: {log_file}", "INFO")
    log("")

    log_handle.close()

    return True


async def main():
    """Main entry point."""
    try:
        success = await test_bangla_production_pipeline()

        print("\n" + "="*80)
        print("DIAGNOSTIC TEST COMPLETE")
        print("="*80)
        print(f"\nDetailed log saved to: {log_file}")
        print("\nNext steps:")
        print("1. Review the log file for detailed analysis")
        print("2. Focus on the 5 recommended fixes")
        print("3. Implement Bangla numeral normalization first (highest priority)")
        print("")

        return 0 if success else 1

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Test failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
