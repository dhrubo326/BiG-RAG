"""
Build Knowledge Graph with OpenAI Embeddings
Modified from script_build.py to use text-embedding-3-large
Follows the exact same pattern as the original script_build.py
"""
import os
import json
import time
from bigrag import BiGRAG
import argparse
import numpy as np
import faiss

# Load OpenAI API key
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

def extract_knowledge(rag, unique_contexts):
    """
    Extract knowledge and build bipartite graph using BiGRAG
    This follows the EXACT pattern from script_build.py
    """
    print(f"Total insert rounds: {len(unique_contexts)//50 + 1}")
    for i in range(0, len(unique_contexts), 50):
        print(f"This is the {i//50 + 1} round of insertion, remain rounds: {len(unique_contexts)//50 - i//50}")
        retries = 0
        max_retries = 50
        while retries < max_retries:
            try:
                rag.insert(unique_contexts[i:i+50])
                break
            except Exception as e:
                retries += 1
                print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
                time.sleep(10)
        if retries == max_retries:
            print("Insertion failed after exceeding the maximum number of retries")


    retries = 0
    max_retries = 50
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")

def embed_knowledge(data_source):
    """
    Create embeddings using OpenAI text-embedding-3-large
    Modified from original to use OpenAI instead of BGE
    """
    # Import OpenAI embedding wrapper
    from bigrag.openai_embedding import OpenAIEmbedding

    # Load text chunks corpus
    corpus = []
    with open(f"expr/{data_source}/kv_store_text_chunks.json", encoding='utf-8') as f:
        texts = json.load(f)
        for item in texts:
            corpus.append(texts[item]['content'])

    # Load entities corpus
    corpus_entity = []
    corpus_entity_des = []
    with open(f"expr/{data_source}/kv_store_entities.json", encoding='utf-8') as f:
        entities = json.load(f)
        for item in entities:
            corpus_entity.append(entities[item]['entity_name'])
            corpus_entity_des.append(entities[item]['content'])

    # Load bipartite edges corpus
    corpus_bipartite_edge = []
    with open(f"expr/{data_source}/kv_store_bipartite_edges.json", encoding='utf-8') as f:
        bipartite_edges = json.load(f)
        for item in bipartite_edges:
            corpus_bipartite_edge.append(bipartite_edges[item]['content'])

    print("\n" + "="*80)
    print("Creating embeddings with OpenAI text-embedding-3-large")
    print("="*80)

    # Initialize OpenAI embedding model
    model = OpenAIEmbedding(
        model_name='text-embedding-3-large',
        dimensions=3072
    )

    # Embed text chunks
    print(f"\n[1/4] Embedding {len(corpus)} text chunks...")
    embeddings = model.encode_corpus(corpus, batch_size=50)
    np.save(f"expr/{data_source}/corpus.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus.npy")
    dim = corpus_numpy.shape[-1]
    corpus_numpy = corpus_numpy.astype(np.float32)

    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index.bin")
    print(f"   ✅ Text chunks index saved: {len(corpus)} vectors ({dim} dimensions)")

    # Embed entities (using descriptions)
    print(f"\n[2/4] Embedding {len(corpus_entity_des)} entity descriptions...")
    embeddings = model.encode_corpus(corpus_entity_des, batch_size=50)
    np.save(f"expr/{data_source}/corpus_entity.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus_entity.npy")
    dim = corpus_numpy.shape[-1]
    corpus_numpy = corpus_numpy.astype(np.float32)

    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index_entity.bin")
    print(f"   ✅ Entity index saved: {len(corpus_entity)} vectors ({dim} dimensions)")

    # Embed bipartite edges
    print(f"\n[3/4] Embedding {len(corpus_bipartite_edge)} bipartite edges...")
    embeddings = model.encode_corpus(corpus_bipartite_edge, batch_size=50)
    np.save(f"expr/{data_source}/corpus_bipartite_edge.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus_bipartite_edge.npy")
    dim = corpus_numpy.shape[-1]
    corpus_numpy = corpus_numpy.astype(np.float32)

    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index_bipartite_edge.bin")
    print(f"   ✅ Bipartite edge index saved: {len(corpus_bipartite_edge)} vectors ({dim} dimensions)")

def insert_knowledge(data_source, unique_contexts):
    """
    Main function: extract knowledge + embed knowledge
    Follows EXACT pattern from script_build.py
    """
    print("\n" + "="*80)
    print(f"PHASE 1: Extracting Knowledge for {data_source}")
    print("="*80)

    rag = BiGRAG(
        working_dir=f"expr/{data_source}"
    )
    extract_knowledge(rag, unique_contexts)

    print("\n" + "="*80)
    print(f"PHASE 2: Creating Embeddings for {data_source}")
    print("="*80)

    embed_knowledge(data_source)

    print("\n" + "="*80)
    print(f"✅ BUILD COMPLETE!")
    print("="*80)
    print(f"Knowledge successfully inserted and embedded for {data_source}")
    print(f"Output directory: expr/{data_source}")
    print(f"\nNext step: Deploy the RAG API (Phase 7)")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, default="MyEducationRAG")
    args = parser.parse_args()
    data_source = args.data_source

    # Read corpus
    unique_contexts = []
    corpus_path = f"datasets/{data_source}/corpus.jsonl"

    if not os.path.exists(corpus_path):
        print(f"❌ Error: {corpus_path} not found!")
        print("Please run Phase 5 first to create the corpus file.")
        exit(1)

    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            unique_contexts.append(data["contents"])

    print("="*80)
    print(f"Building Knowledge Graph for: {data_source}")
    print(f"Using OpenAI text-embedding-3-large (3072 dimensions)")
    print(f"Using OpenAI GPT-4o for entity extraction")
    print("="*80)
    print(f"Loaded {len(unique_contexts)} documents from corpus.jsonl")
    print(f"Expected time: 30-45 minutes")
    print(f"Expected cost: ~$1-3 USD")
    print("="*80)

    insert_knowledge(data_source, unique_contexts)
