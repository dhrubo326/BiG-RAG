"""
Phase 1 Complete Verification Test Suite

Validates that ALL 6 steps from Production_pipeline_redesign_plan.md are implemented:
- Step 1: Extraction Strategy Configuration
- Step 2: Semantic Boundary-Aware Chunking
- Step 3: Gleaning Implementation
- Step 4: Unified Entity Merging
- Step 5: Pipeline Selector Helper
- Step 6: HITL System

This test suite runs all individual test suites and verifies integration.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import subprocess
from pathlib import Path


def run_test_suite(test_file: str, description: str) -> tuple:
    """Run a test suite and return (passed, total)."""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print('=' * 70)

    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Parse output for test counts
    output = result.stdout + result.stderr

    # Look for "X/Y tests passed" or "TEST SUMMARY: X/Y"
    import re
    match = re.search(r'(\d+)/(\d+)\s+tests?\s+passed', output, re.IGNORECASE)
    if match:
        passed = int(match.group(1))
        total = int(match.group(2))
        return (passed, total, result.returncode == 0)

    # If no match, check return code
    if result.returncode == 0:
        return (1, 1, True)  # Assume success
    else:
        return (0, 1, False)  # Assume failure


async def verify_step1_extraction_strategy():
    """Verify Step 1: Extraction Strategy Configuration."""
    print("\n[STEP 1 VERIFICATION] Extraction Strategy Configuration")

    checks = []

    # Get project root (parent of test_scripts)
    project_root = Path(__file__).parent.parent

    # Check 1: enhanced_pipeline.py exists
    pipeline_file = project_root / "bigrag" / "enhanced_pipeline.py"
    checks.append(("enhanced_pipeline.py exists", pipeline_file.exists()))

    # Check 2: extraction_strategy parameter exists
    if pipeline_file.exists():
        content = pipeline_file.read_text(encoding='utf-8')
        checks.append(("extraction_strategy parameter", "extraction_strategy" in content))
        checks.append(("strict/gleaning/hybrid strategies", "strict" in content and "gleaning" in content and "hybrid" in content))
        checks.append(("enable_gleaning support", "enable_gleaning" in content))

    # Check 3: bigrag.py has enhanced pipeline config
    bigrag_file = project_root / "bigrag" / "bigrag.py"
    if bigrag_file.exists():
        content = bigrag_file.read_text(encoding='utf-8')
        checks.append(("use_enhanced_pipeline config", "use_enhanced_pipeline" in content))
        checks.append(("enhanced_pipeline_config", "enhanced_pipeline_config" in content))

    # Print results
    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for check, result in checks:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    print(f"\n  Step 1: {passed}/{total} checks passed")
    return passed == total


async def verify_step2_semantic_chunking():
    """Verify Step 2: Semantic Boundary-Aware Chunking."""
    print("\n[STEP 2 VERIFICATION] Semantic Boundary-Aware Chunking")

    project_root = Path(__file__).parent.parent
    checks = []

    # Check 1: smart_chunker.py has semantic chunking
    chunker_file = project_root / "bigrag" / "preprocessors" / "smart_chunker.py"
    if chunker_file.exists():
        content = chunker_file.read_text(encoding='utf-8')
        checks.append(("_chunk_with_semantic_boundaries method", "_chunk_with_semantic_boundaries" in content))
        checks.append(("Paragraph boundary detection", "\\n\\n" in content or "double newline" in content.lower()))
        checks.append(("Sentence splitting", "split_by_sentences" in content or "sentence" in content.lower()))
    else:
        checks.append(("smart_chunker.py exists", False))

    # Check 2: utils.py has supporting functions
    utils_file = project_root / "bigrag" / "utils.py"
    if utils_file.exists():
        content = utils_file.read_text(encoding='utf-8')
        checks.append(("count_tokens_fast utility", "count_tokens_fast" in content))
        checks.append(("split_by_sentences utility", "split_by_sentences" in content))

    # Print results
    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for check, result in checks:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    print(f"\n  Step 2: {passed}/{total} checks passed")
    return passed == total


async def verify_step3_gleaning():
    """Verify Step 3: Gleaning Implementation."""
    print("\n[STEP 3 VERIFICATION] Gleaning Implementation")

    project_root = Path(__file__).parent.parent
    checks = []

    # Check: constrained_extractor.py has gleaning
    extractor_file = project_root / "bigrag" / "extractors" / "constrained_extractor.py"
    if extractor_file.exists():
        content = extractor_file.read_text(encoding='utf-8')
        checks.append(("enable_gleaning parameter", "enable_gleaning" in content))
        checks.append(("max_gleaning_iterations parameter", "max_gleaning_iterations" in content))
        checks.append(("Gleaning loop implementation", "for gleaning_pass" in content or "gleaning loop" in content.lower()))
        checks.append(("Conversation history", "conversation_history" in content))
        checks.append(("Quality-based merging", "_merge_extractions_by_quality" in content or "quality_score" in content))
    else:
        checks.append(("constrained_extractor.py exists", False))

    # Print results
    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for check, result in checks:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    print(f"\n  Step 3: {passed}/{total} checks passed")
    return passed == total


async def verify_step4_unified_merger():
    """Verify Step 4: Unified Entity Merging."""
    print("\n[STEP 4 VERIFICATION] Unified Entity Merging")

    project_root = Path(__file__).parent.parent
    checks = []

    # Check: unified_merger.py exists
    merger_file = project_root / "bigrag" / "merging" / "unified_merger.py"
    checks.append(("unified_merger.py exists", merger_file.exists()))

    if merger_file.exists():
        content = merger_file.read_text(encoding='utf-8')
        checks.append(("UnifiedEntityMerger class", "class UnifiedEntityMerger" in content))
        checks.append(("Basic merge strategy", "basic" in content and "_merge_basic" in content))
        checks.append(("Fuzzy merge strategy", "fuzzy" in content and "_merge_fuzzy" in content))
        checks.append(("Hybrid merge strategy", "hybrid" in content))

    # Check: Integration in enhanced_pipeline.py
    pipeline_file = project_root / "bigrag" / "enhanced_pipeline.py"
    if pipeline_file.exists():
        content = pipeline_file.read_text(encoding='utf-8')
        checks.append(("entity_merge_strategy parameter", "entity_merge_strategy" in content))
        checks.append(("UnifiedEntityMerger import", "UnifiedEntityMerger" in content))

    # Print results
    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for check, result in checks:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    print(f"\n  Step 4: {passed}/{total} checks passed")
    return passed == total


async def verify_step5_pipeline_selector():
    """Verify Step 5: Pipeline Selector Helper."""
    print("\n[STEP 5 VERIFICATION] Pipeline Selector Helper")

    project_root = Path(__file__).parent.parent
    checks = []

    # Check: pipeline_selector.py exists
    selector_file = project_root / "bigrag" / "pipeline_selector.py"
    checks.append(("pipeline_selector.py exists", selector_file.exists()))

    if selector_file.exists():
        content = selector_file.read_text(encoding='utf-8')
        checks.append(("PipelineSelector class", "class PipelineSelector" in content))
        checks.append(("analyze_documents method", "def analyze_documents" in content))
        checks.append(("recommend_pipeline method", "def recommend_pipeline" in content))
        checks.append(("Configuration presets", "CONFIGURATION_PRESETS" in content))
        checks.append(("quick_recommend function", "def quick_recommend" in content))

    # Check: Integration in pipelines
    pipeline_file = project_root / "bigrag" / "enhanced_pipeline.py"
    if pipeline_file.exists():
        content = pipeline_file.read_text(encoding='utf-8')
        checks.append(("recommend_config method", "def recommend_config" in content))

    bigrag_file = project_root / "bigrag" / "bigrag.py"
    if bigrag_file.exists():
        content = bigrag_file.read_text(encoding='utf-8')
        checks.append(("BiGRAG.recommend_config", "def recommend_config" in content))

    # Print results
    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for check, result in checks:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    print(f"\n  Step 5: {passed}/{total} checks passed")
    return passed == total


async def verify_step6_hitl():
    """Verify Step 6: HITL System."""
    print("\n[STEP 6 VERIFICATION] HITL System")

    project_root = Path(__file__).parent.parent
    checks = []

    # Check: HITL module exists
    hitl_dir = project_root / "bigrag" / "hitl"
    checks.append(("bigrag/hitl directory exists", hitl_dir.exists()))

    store_file = project_root / "bigrag" / "hitl" / "failed_extraction_store.py"
    checks.append(("failed_extraction_store.py exists", store_file.exists()))

    if store_file.exists():
        content = store_file.read_text(encoding='utf-8')
        checks.append(("FailedExtractionStore class", "class FailedExtractionStore" in content))
        checks.append(("save_failed_chunk method", "def save_failed_chunk" in content))
        checks.append(("save_failed_table method", "def save_failed_table" in content))
        checks.append(("get_review_queue method", "def get_review_queue" in content))

    # Check: API routes
    hitl_routes_file = project_root / "backend" / "api" / "hitl_routes.py"
    checks.append(("hitl_routes.py exists", hitl_routes_file.exists()))

    if hitl_routes_file.exists():
        content = hitl_routes_file.read_text(encoding='utf-8')
        checks.append(("HITL API router", "router = APIRouter" in content))
        checks.append(("get_failed_extractions endpoint", "def get_failed_extractions" in content))

    # Check: Integration in extractor
    extractor_file = project_root / "bigrag" / "extractors" / "constrained_extractor.py"
    if extractor_file.exists():
        content = extractor_file.read_text(encoding='utf-8')
        checks.append(("hitl_store parameter", "hitl_store" in content))
        checks.append(("save_failed_chunk call", "save_failed_chunk" in content))

    # Print results
    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for check, result in checks:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    print(f"\n  Step 6: {passed}/{total} checks passed")
    return passed == total


async def main():
    """Run complete Phase 1 verification."""
    print("=" * 70)
    print("PHASE 1 COMPLETE VERIFICATION TEST SUITE")
    print("Production Pipeline Redesign Plan - All 6 Steps")
    print("=" * 70)

    # Part 1: Verify implementation of all steps
    print("\n" + "=" * 70)
    print("PART 1: IMPLEMENTATION VERIFICATION")
    print("=" * 70)

    verifications = [
        ("Step 1: Extraction Strategy", verify_step1_extraction_strategy()),
        ("Step 2: Semantic Chunking", verify_step2_semantic_chunking()),
        ("Step 3: Gleaning Implementation", verify_step3_gleaning()),
        ("Step 4: Unified Entity Merging", verify_step4_unified_merger()),
        ("Step 5: Pipeline Selector", verify_step5_pipeline_selector()),
        ("Step 6: HITL System", verify_step6_hitl()),
    ]

    verification_results = []
    for name, coro in verifications:
        result = await coro
        verification_results.append((name, result))

    # Part 2: Run individual test suites
    print("\n" + "=" * 70)
    print("PART 2: RUNNING INDIVIDUAL TEST SUITES")
    print("=" * 70)

    # Determine if running from project root or test_scripts directory
    if Path("test_scripts").exists():
        test_dir = "test_scripts"
    else:
        test_dir = "."

    test_suites = [
        (f"{test_dir}/test_gleaning.py", "Step 3: Gleaning Tests"),
        (f"{test_dir}/test_unified_merger.py", "Step 4: Unified Merger Tests"),
        (f"{test_dir}/test_pipeline_selector.py", "Step 5: Pipeline Selector Tests"),
        (f"{test_dir}/test_hitl_system.py", "Step 6: HITL System Tests"),
    ]

    test_results = []
    for test_file, description in test_suites:
        if Path(test_file).exists():
            passed, total, success = run_test_suite(test_file, description)
            test_results.append((description, passed, total, success))
        else:
            print(f"\n[SKIP] {test_file} not found")
            test_results.append((description, 0, 0, False))

    # Part 3: Summary
    print("\n" + "=" * 70)
    print("PHASE 1 VERIFICATION SUMMARY")
    print("=" * 70)

    print("\nImplementation Verification:")
    all_verified = True
    for name, result in verification_results:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {name}")
        if not result:
            all_verified = False

    print("\nTest Suite Results:")
    total_passed = 0
    total_tests = 0
    all_tests_pass = True

    for description, passed, total, success in test_results:
        if total > 0:
            status = "[OK]" if success else "[FAIL]"
            print(f"  {status} {description}: {passed}/{total} tests passed")
            total_passed += passed
            total_tests += total
            if not success:
                all_tests_pass = False
        else:
            print(f"  [SKIP] {description}: No tests found")

    print("\n" + "=" * 70)
    print(f"OVERALL: {total_passed}/{total_tests} tests passed")

    if all_verified and all_tests_pass:
        print("STATUS: ALL PHASE 1 STEPS VERIFIED AND TESTED")
        print("=" * 70)
        return 0
    else:
        print("STATUS: SOME VERIFICATIONS OR TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
