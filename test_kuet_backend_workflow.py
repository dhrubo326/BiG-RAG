"""
Test Script: KUET Indexing Using EXACT Backend Workflow

This replicates the EXACT workflow used by the backend API:
1. Save document to datasets/demo_test/raw/corpus.jsonl
2. Call BiGRAG.ainsert() (same as backend does)
3. Output to expr/demo_test/
4. Inspect generated files

This is the PROPER way to test before integration.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bigrag import BiGRAG
from bigrag.utils import logger


async def add_document_to_corpus(
    data_source: str,
    doc_id: str,
    content: str,
    title: str,
    metadata: dict = None
):
    """
    Add document to corpus.jsonl (same as backend API does).

    This is Step 1 of the backend workflow.
    """
    print(f"\n[STEP 1/4] Adding document to corpus...")

    PROJECT_ROOT = Path(__file__).parent
    corpus_file = PROJECT_ROOT / "datasets" / data_source / "raw" / "corpus.jsonl"

    # Create directory if doesn't exist
    corpus_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare document (same format as backend)
    doc = {
        "id": doc_id,
        "contents": content,
        "title": title,
        "upload_date": datetime.now().isoformat(),
        "source": "test_script"
    }

    # Add metadata if provided
    if metadata:
        doc["metadata"] = metadata

    # Append to corpus
    with open(corpus_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"[OK] Document added to corpus")
    print(f"     Corpus file: {corpus_file}")
    print(f"     Document ID: {doc_id}")
    print(f"     Title: {title}")

    return corpus_file


async def process_with_bigrag(
    content: str,
    title: str,
    data_source: str,
    metadata: dict = None
):
    """
    Process document with BiGRAG (same as backend API does).

    This is Step 2 of the backend workflow.
    """
    print(f"\n[STEP 2/4] Processing with BiGRAG...")

    # Read API key and set environment variable (BiGRAG reads from env)
    api_key_file = "openai_api_key.txt"
    if not os.path.exists(api_key_file):
        print(f"[ERROR] {api_key_file} not found!")
        return None

    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()

    # Set OpenAI API key in environment (BiGRAG LLM functions read from here)
    os.environ['OPENAI_API_KEY'] = api_key

    # Initialize BiGRAG (EXACT same settings as backend)
    working_dir = f"expr/{data_source}"

    print(f"[INFO] Initializing BiGRAG...")
    print(f"       Working dir: {working_dir}")
    print(f"       Chunk size: 1200 tokens")
    print(f"       Overlap: 100 tokens")
    print(f"       Model: gpt-4o-mini (for entity extraction)")

    rag = BiGRAG(
        working_dir=working_dir,
        enable_llm_cache=True,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        tiktoken_model_name="gpt-4o"
    )

    # Prepare metadata (same as backend)
    doc_metadata = metadata or {}
    if title and "title" not in doc_metadata:
        doc_metadata["title"] = title

    # Insert document (this is what backend calls at line 150 in jobs.py)
    print(f"\n[INFO] Calling BiGRAG.ainsert()...")
    start_time = datetime.now()

    await rag.ainsert(content, metadata=doc_metadata)

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    print(f"[OK] Processing completed in {processing_time:.2f} seconds")

    # Get statistics (same as backend)
    # Note: chunk_entity_relation_graph is NetworkXStorage, need to access internal _graph
    graph_storage = rag.chunk_entity_relation_graph
    graph = graph_storage._graph if hasattr(graph_storage, '_graph') else graph_storage

    entity_nodes = [n for n, d in graph.nodes(data=True) if d.get('role') == 'entity']
    relation_nodes = [n for n, d in graph.nodes(data=True) if d.get('role') == 'relation']
    chunk_keys = await rag.text_chunks.all_keys()
    chunks = await rag.text_chunks.get_by_ids(chunk_keys)

    stats = {
        'total_entities': len(entity_nodes),
        'total_relations': len(relation_nodes),
        'total_chunks': len(chunks),
        'processing_time': processing_time,
        'working_dir': working_dir
    }

    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total Entities: {stats['total_entities']}")
    print(f"Total Relations: {stats['total_relations']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Output Directory: {working_dir}")
    print("="*80)

    return stats


async def inspect_output_files(data_source: str):
    """
    Inspect generated files (Step 3).
    """
    print(f"\n[STEP 3/4] Inspecting output files...")

    working_dir = Path(f"expr/{data_source}")

    if not working_dir.exists():
        print(f"[ERROR] Output directory not found: {working_dir}")
        return False

    # List all files
    files = list(working_dir.glob("*"))

    print(f"\n[INFO] Files in {working_dir}:")
    for f in sorted(files):
        if f.is_file():
            size = f.stat().st_size
            print(f"       - {f.name} ({size:,} bytes)")

    # Check critical files
    critical_files = [
        "graph_chunk_entity_relation.graphml",
        "vdb_entities.json",
        "vdb_relations.json",  # Note: Renamed from vdb_bipartite_edges.json
        "vdb_chunks.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json"
    ]

    print(f"\n[INFO] Critical files check:")
    all_present = True
    for filename in critical_files:
        filepath = working_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            status = "OK" if size > 0 else "EMPTY"
            print(f"       [{status}] {filename} ({size:,} bytes)")
        else:
            print(f"       [MISSING] {filename}")
            all_present = False

    if all_present:
        print(f"\n[OK] All critical files present")
    else:
        print(f"\n[WARN] Some critical files missing")

    return all_present


async def save_test_results(stats, data_source):
    """
    Save test results (Step 4).
    """
    print(f"\n[STEP 4/4] Saving test results...")

    test_results = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'backend_workflow_replication',
        'document': 'KUET_Admission_info.md',
        'data_source': data_source,
        'workflow_steps': [
            '1. add_document_to_corpus() - Save to datasets/demo_test/raw/corpus.jsonl',
            '2. BiGRAG.ainsert() - Process with OLD chunking-first approach',
            '3. Output to expr/demo_test/',
            '4. Inspect generated files'
        ],
        'statistics': stats,
        'corpus_location': f'datasets/{data_source}/raw/corpus.jsonl',
        'graph_location': f'expr/{data_source}/'
    }

    results_file = "test_results_kuet_backend.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Test results saved to: {results_file}")


async def main():
    """
    Main test execution - replicates EXACT backend workflow.
    """

    print("="*80)
    print("KUET INDEXING TEST - BACKEND WORKFLOW REPLICATION")
    print("="*80)
    print("This test replicates EXACTLY what the backend API does:")
    print("  1. Save to datasets/demo_test/raw/corpus.jsonl")
    print("  2. Call BiGRAG.ainsert() with OLD approach")
    print("  3. Output to expr/demo_test/")
    print("  4. Files can be loaded in backend API")
    print("="*80)

    # Configuration
    data_source = "demo_test"
    doc_id = "kuet_admission_2024_25"
    title = "KUET Admission Information 2024-2025"

    # Load KUET document
    kuet_file = Path("KUET_Admission_info.md")
    if not kuet_file.exists():
        print(f"\n[ERROR] Document not found: {kuet_file}")
        return

    with open(kuet_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"\n[INFO] Loaded document: {kuet_file}")
    print(f"       Length: {len(content)} characters")

    metadata = {
        "category": "admission",
        "university": "KUET",
        "tags": ["kuet", "admission", "engineering", "undergraduate"],
        "year": "2024-2025",
        "source": "KUET_Admission_info.md"
    }

    # Execute workflow
    try:
        # Step 1: Add to corpus
        corpus_file = await add_document_to_corpus(
            data_source=data_source,
            doc_id=doc_id,
            content=content,
            title=title,
            metadata=metadata
        )

        # Step 2: Process with BiGRAG
        stats = await process_with_bigrag(
            content=content,
            title=title,
            data_source=data_source,
            metadata=metadata
        )

        if not stats:
            print("\n[ERROR] Processing failed - stopping")
            return

        # Step 3: Inspect files
        files_ok = await inspect_output_files(data_source)

        # Step 4: Save results
        await save_test_results(stats, data_source)

        # Final summary
        print("\n" + "="*80)
        print("TEST COMPLETION SUMMARY")
        print("="*80)
        print(f"[OK] Test completed successfully")
        print(f"\nGenerated files:")
        print(f"  - Corpus: datasets/{data_source}/raw/corpus.jsonl")
        print(f"  - Graph: expr/{data_source}/graph_chunk_entity_relation.graphml")
        print(f"  - Vector DBs: expr/{data_source}/vdb_*.json")
        print(f"  - KV stores: expr/{data_source}/kv_store_*.json")
        print(f"\nYou can now:")
        print(f"  1. Start backend: cd backend && python server.py --data_source {data_source}")
        print(f"  2. Start frontend: cd frontend && npm run dev")
        print(f"  3. View documents at: http://localhost:3000/documents")
        print(f"  4. Test queries at: http://localhost:3000/")
        print("="*80)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
