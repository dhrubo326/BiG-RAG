"""Verify vector database content in detail."""
import json
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("VECTOR DATABASE VERIFICATION")
print("=" * 80)

# Check each VDB file
vdb_files = {
    'entities': r'D:\BiG-RAG\expr\bangla_diagnosis_test\vdb_entities.json',
    'relations': r'D:\BiG-RAG\expr\bangla_diagnosis_test\vdb_relations.json',
    'chunks': r'D:\BiG-RAG\expr\bangla_diagnosis_test\vdb_chunks.json'
}

for name, path in vdb_files.items():
    print(f"\n[{name.upper()}]")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    print(f"File keys: {list(data.keys())}")
    print(f"Embedding dimension: {data.get('embedding_dim', 'N/A')}")

    # Check data
    data_list = data.get('data', [])
    print(f"Data items: {len(data_list)}")
    if len(data_list) > 0:
        print(f"Sample data (first 3):")
        for i, item in enumerate(data_list[:3]):
            if isinstance(item, dict):
                print(f"  {i+1}. Keys: {list(item.keys())}")
            else:
                print(f"  {i+1}. Type: {type(item)}, Value: {str(item)[:80]}")

    # Check matrix
    matrix = data.get('matrix', [])
    if isinstance(matrix, str):
        print(f"Matrix stored as string, length: {len(matrix)}")
    elif matrix and len(matrix) > 0:
        print(f"Matrix has {len(matrix)} items")
        if isinstance(matrix[0], list):
            print(f"Matrix is 2D list with shape: ({len(matrix)}, {len(matrix[0]) if matrix else 0})")
        else:
            print(f"Matrix is 1D list/array")
    else:
        print("Matrix: EMPTY or invalid")

print("\n" + "=" * 80)
