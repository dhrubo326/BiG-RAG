"""
Quick script to rebuild SingleTopic graph with embeddings
"""
import asyncio
from bigrag import BiGRAG

async def rebuild():
    print("Rebuilding SingleTopic knowledge graph...")

    # Initialize BiGRAG
    rag = BiGRAG(
        working_dir="./expr/SingleTopic",
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )

    # Read the existing documents
    import json
    with open("expr/SingleTopic/kv_store_full_docs.json", 'r') as f:
        docs = json.load(f)

    print(f"Found {len(docs)} documents to reindex...")

    # Extract text from documents
    texts = []
    metadata = []
    for doc in docs:
        if isinstance(doc, dict):
            texts.append(doc.get('content', ''))
            metadata.append({
                'title': doc.get('title', 'Untitled'),
                'source': doc.get('source', 'unknown')
            })

    if texts:
        print(f"Reindexing {len(texts)} documents...")
        # Re-insert to build proper indices
        await rag.ainsert(texts, metadata=metadata)
        print("Graph rebuilt successfully!")
    else:
        print("No valid documents found to index")

if __name__ == "__main__":
    asyncio.run(rebuild())