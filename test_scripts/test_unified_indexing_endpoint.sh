#!/bin/bash
# Test Unified Indexing Endpoint with Different Feature Combinations

BASE_URL="http://localhost:8001"
TEST_FILE="../example_kuet_mini.md"

echo "=========================================="
echo "Testing Unified Indexing Endpoint"
echo "=========================================="

# Test 1: Minimal Configuration (Fast & Cheap)
echo ""
echo "[Test 1] Minimal Configuration (like Standard preset)"
echo "------------------------------------------"
curl -X POST "$BASE_URL/indexing/index-document" \
  -F "file=@$TEST_FILE" \
  -F "data_source=test_minimal" \
  -F "title=Test Minimal Config" \
  -F "process_async=false" \
  | jq '.'

echo ""
echo ""

# Test 2: High Quality Configuration (like Quality preset)
echo "[Test 2] High Quality Configuration (like Quality preset)"
echo "------------------------------------------"
curl -X POST "$BASE_URL/indexing/index-document" \
  -F "file=@$TEST_FILE" \
  -F "data_source=test_quality" \
  -F "title=Test Quality Config" \
  -F "need_table_extraction=true" \
  -F "need_dynamic_chunking=true" \
  -F "need_gleaning=true" \
  -F "gleaning_iterations=2" \
  -F "need_table_fact_extraction=true" \
  -F "need_numeric_validation=true" \
  -F "need_semantic_validation=true" \
  -F "merge_strategy=fuzzy" \
  -F "enable_hitl=true" \
  -F "enable_orphan_linking=true" \
  -F "process_async=false" \
  | jq '.'

echo ""
echo ""

# Test 3: Custom Mix (Tables + Numeric Validation Only)
echo "[Test 3] Custom Mix (Tables + Numeric Validation)"
echo "------------------------------------------"
curl -X POST "$BASE_URL/indexing/index-document" \
  -F "file=@$TEST_FILE" \
  -F "data_source=test_custom" \
  -F "title=Test Custom Mix" \
  -F "need_table_extraction=true" \
  -F "need_table_fact_extraction=true" \
  -F "need_numeric_validation=true" \
  -F "merge_strategy=basic" \
  -F "process_async=false" \
  | jq '.'

echo ""
echo ""

# Test 4: Balanced Configuration
echo "[Test 4] Balanced Configuration"
echo "------------------------------------------"
curl -X POST "$BASE_URL/indexing/index-document" \
  -F "file=@$TEST_FILE" \
  -F "data_source=test_balanced" \
  -F "title=Test Balanced Config" \
  -F "need_dynamic_chunking=true" \
  -F "need_gleaning=true" \
  -F "gleaning_iterations=1" \
  -F "need_semantic_validation=true" \
  -F "merge_strategy=basic" \
  -F "process_async=false" \
  | jq '.'

echo ""
echo "=========================================="
echo "All tests complete!"
echo "=========================================="
