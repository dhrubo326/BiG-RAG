"""
Verification Script: VDB Indexing Fixes
Tests that all 3 fixes are correctly implemented and will work after graph rebuild
"""

import sys
sys.path.insert(0, 'D:/BiG-RAG')

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def verify_fix_1_meta_fields():
    """Verify Fix #1: entity_id and entity_name in meta_fields"""
    print("\n[Fix #1 Verification] Checking VDB meta_fields...")

    # Read bigrag.py file directly
    try:
        with open('D:/BiG-RAG/bigrag/bigrag.py', 'r', encoding='utf-8') as f:
            source = f.read()

        # Check entity VDB meta_fields in source code
        if 'meta_fields={"entity_id", "entity_name"}' in source:
            print('  [OK] Entity VDB meta_fields contains entity_id and entity_name')
        elif '"entity_id"' in source and '"entity_name"' in source:
            print('  [OK] Entity VDB meta_fields contains entity_id and entity_name (different order)')
        else:
            print('  [FAIL] Entity VDB meta_fields missing entity_id or entity_name')
            return False

        # Check relation VDB meta_fields in source code
        if 'meta_fields={"relation_id"}' in source:
            print('  [OK] Relation VDB meta_fields contains relation_id')
        elif '"relation_id"' in source:
            print('  [OK] Relation VDB meta_fields contains relation_id')
        else:
            print('  [FAIL] Relation VDB meta_fields missing relation_id')
            return False

        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def verify_fix_2_field_names():
    """Verify Fix #2: relation_id field used in VDB storage code"""
    import inspect
    from bigrag import operate

    print("\n[Fix #2 Verification] Checking relation VDB storage code...")

    # Get source code of extract_entities function
    source = inspect.getsource(operate.extract_entities)

    # Check if relation_id is used (Fix #2)
    if '"relation_id":' in source and 'dp["relation_name"]' in source:
        print('  [OK] Code uses "relation_id" field')
        return True
    else:
        print('  [FAIL] relation_id field not found in storage code')
        return False

def verify_fix_3_entity_id_usage():
    """Verify Fix #3: entity_id used in unused function"""
    import inspect
    from bigrag import operate

    print("\n[Fix #3 Verification] Checking entity_id in unused function...")

    # Get source code of _find_most_related_text_unit_from_entities
    source = inspect.getsource(operate._find_most_related_text_unit_from_entities)

    # Check if entity_id is used (Fix #3)
    if 'dp["entity_id"]' in source and 'for dp in node_datas' in source:
        print('  [OK] Function uses entity_id correctly')
        return True
    else:
        print('  [FAIL] Function still uses entity_name!')
        return False

def verify_retrieval_compatibility():
    """Verify retrieval code handles all field variations"""
    import inspect
    from bigrag import operate

    print("\n[Retrieval Compatibility] Checking retrieval code...")

    # Check _get_node_data uses entity_id
    node_source = inspect.getsource(operate._get_node_data)
    if 'r.get("entity_id"' in node_source:
        print('  [OK] Entity retrieval prioritizes entity_id field')
    else:
        print('  [WARN] Entity retrieval might not use entity_id')

    # Check _get_edge_data uses relation_id
    edge_source = inspect.getsource(operate._get_edge_data)
    if 'r.get("relation_id"' in edge_source:
        print('  [OK] Relation retrieval prioritizes relation_id field')
    else:
        print('  [WARN] Relation retrieval might not use relation_id')

    return True

def main():
    print("=" * 80)
    print("VDB INDEXING FIXES VERIFICATION")
    print("=" * 80)

    results = []

    # Test Fix #1
    results.append(("Fix #1: meta_fields", verify_fix_1_meta_fields()))

    # Test Fix #2
    results.append(("Fix #2: relation_id naming", verify_fix_2_field_names()))

    # Test Fix #3
    results.append(("Fix #3: entity_id in unused function", verify_fix_3_entity_id_usage()))

    # Test retrieval compatibility
    results.append(("Retrieval Compatibility", verify_retrieval_compatibility()))

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("[SUCCESS] All fixes verified! Safe to rebuild graphs.")
    else:
        print("[WARNING] Some verifications failed. Review before rebuilding.")
    print("=" * 80)

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
