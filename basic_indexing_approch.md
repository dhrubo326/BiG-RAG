🔄 COMPLETE DOCUMENT LIFECYCLE
Phase 1: Document Upload & Deduplication
# graphr1/graphr1.py lines 272-286
async def ainsert(self, string_or_strings):
    # Step 1: Convert to list
    if isinstance(string_or_strings, str):
        string_or_strings = [string_or_strings]
    
    # Step 2: Generate unique IDs using MD5 hash
    new_docs = {
        compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
        for c in string_or_strings
    }
    
    # Step 3: Deduplication - Filter out already indexed documents
    _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
    new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
    
    if not len(new_docs):
        logger.warning("All docs are already in the storage")
        return
What's happening:
✅ Content-based ID: Uses MD5 hash of document content → Same content = Same ID
✅ Automatic deduplication: Skips documents already in corpus
✅ Idempotent: Uploading same document twice = no duplicates
ID Format: doc-{md5_hash_of_content}
Example: doc-5d41402abc4b2a76b9719d911017c592
Phase 2: Document Chunking
# graphr1/graphr1.py lines 290-316
inserting_chunks = {}
for doc_key, doc in tqdm_async(new_docs.items(), desc="Chunking documents"):
    chunks = {
        compute_mdhash_id(dp["content"], prefix="chunk-"): {
            **dp,
            "full_doc_id": doc_key,  # Link back to original document
        }
        for dp in chunking_by_token_size(
            doc["content"],
            overlap_token_size=self.chunk_overlap_token_size,  # Default: 100 tokens
            max_token_size=self.chunk_token_size,              # Default: 1200 tokens
            tiktoken_model=self.tiktoken_model_name,
        )
    }
    inserting_chunks.update(chunks)

# Deduplication at chunk level
_add_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
Chunking Strategy:
# graphr1/operate.py lines 35-53
def chunking_by_token_size(content, overlap_token_size=128, max_token_size=1024):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size]
        )
        results.append({
            "tokens": min(max_token_size, len(tokens) - start),
            "content": chunk_content.strip(),
            "chunk_order_index": index,  # Preserves document order
        })
    return results
Visual Example:
Document (5000 tokens)
├─ Chunk 0: tokens [0:1200]      → "chunk-abc123..."
├─ Chunk 1: tokens [1100:2300]   → "chunk-def456..."  (100 overlap with chunk 0)
├─ Chunk 2: tokens [2200:3400]   → "chunk-ghi789..."  (100 overlap with chunk 1)
├─ Chunk 3: tokens [3300:4500]   → "chunk-jkl012..."
└─ Chunk 4: tokens [4400:5000]   → "chunk-mno345..."
Chunk Metadata:
{
  "chunk-abc123...": {
    "content": "The University of Dhaka...",
    "tokens": 1200,
    "chunk_order_index": 0,
    "full_doc_id": "doc-5d41402..."  // Link to parent document
  }
}
Why overlap?
Prevents context loss at chunk boundaries
Entity mentions spanning chunks won't be split
Industry standard: 100-200 token overlap
Phase 3: Entity & Relationship Extraction (LLM-Powered)
# graphr1/graphr1.py lines 318-329
maybe_new_kg = await extract_entities(
    inserting_chunks,
    knowledge_graph_inst=self.chunk_entity_relation_graph,
    entity_vdb=self.entities_vdb,
    hyperedge_vdb=self.hyperedges_vdb,
    global_config=asdict(self),
)
LLM Extraction Process:
# graphr1/operate.py lines 261-481 (simplified)
async def extract_entities(chunks, knowledge_graph_inst, entity_vdb, hyperedge_vdb, global_config):
    for chunk_key, chunk_data in chunks:
        content = chunk_data["content"]
        
        # Step 1: Send chunk to LLM with extraction prompt
        prompt = entity_extract_prompt.format(input_text=content)
        llm_response = await use_llm_func(prompt)
        
        # Step 2: Iterative gleaning (ask LLM "did you miss anything?")
        for i in range(entity_extract_max_gleaning):  # Default: 1
            glean_result = await use_llm_func(continue_prompt, history)
            llm_response += glean_result
            
            # Ask if we should continue
            if_continue = await use_llm_func(if_loop_prompt, history)
            if if_continue != "yes":
                break
        
        # Step 3: Parse LLM output
        records = parse_llm_output(llm_response)
        
        for record in records:
            if record.type == "entity":
                # (entity, "DHAKA UNIVERSITY", "ORGANIZATION", "Premier university", 95)
                entities.append({
                    "entity_name": record.name,
                    "entity_type": record.type,
                    "description": record.description,
                    "weight": record.weight,
                    "source_id": chunk_key  # Link to source chunk
                })
            
            elif record.type == "knowledge-edge":
                # (knowledge-edge, "Dhaka University offers CS programs")
                edges.append({
                    "hyperedge_name": record.statement,
                    "weight": calculate_weight(),
                    "source_id": chunk_key  # Link to source chunk
                })
LLM Prompt Example:
Extract entities and relationships from this text:

The University of Dhaka offers undergraduate programs in Computer Science, 
Mathematics, and Physics through its Faculty of Science.

Output format:
(entity, NAME, TYPE, DESCRIPTION, WEIGHT)
(knowledge-edge, RELATIONSHIP_STATEMENT)

Examples:
(entity, "UNIVERSITY OF DHAKA", "ORGANIZATION", "Premier public university", 95)
(entity, "COMPUTER SCIENCE", "PROGRAM", "Academic program", 85)
(knowledge-edge, "University of Dhaka offers CS, Math, Physics programs")
Entity Deduplication & Merging:
# graphr1/operate.py lines 167-212
async def _merge_nodes_then_upsert(entity_name, nodes_data, knowledge_graph_inst):
    """
    Multiple chunks may mention same entity → Merge descriptions
    """
    # Get existing node
    existing_node = await knowledge_graph_inst.get_node(entity_name)
    
    if existing_node:
        # Merge descriptions
        old_description = existing_node.get("description", "")
        new_descriptions = [node["description"] for node in nodes_data]
        merged_description = old_description + GRAPH_FIELD_SEP + GRAPH_FIELD_SEP.join(new_descriptions)
        
        # Aggregate source_ids
        old_source_ids = set(existing_node.get("source_id", "").split(GRAPH_FIELD_SEP))
        new_source_ids = set([node["source_id"] for node in nodes_data])
        all_source_ids = GRAPH_FIELD_SEP.join(old_source_ids | new_source_ids)
        
        # Summarize if description too long (LLM summarization)
        if len(merged_description) > max_tokens:
            merged_description = await _handle_entity_relation_summary(
                entity_name, merged_description, global_config
            )
    
    # Upsert to graph
    await knowledge_graph_inst.upsert_node(entity_name, {
        "entity_name": entity_name,
        "description": merged_description,
        "source_id": all_source_ids,
        "weight": aggregated_weight
    })
Key insight: Same entity mentioned in multiple chunks → Single graph node with merged descriptions
Phase 4: Bipartite Graph Construction
# Nodes are added first
for entity in entities:
    await _merge_nodes_then_upsert(entity, knowledge_graph)

for edge in hyperedges:
    await _merge_hyperedges_then_upsert(edge, knowledge_graph)

# Then edges connecting hyperedge → entity
for edge_name, entity_names in connections:
    for entity_name in entity_names:
        await knowledge_graph.upsert_edge(
            source=edge_name,   # Knowledge edge node
            target=entity_name, # Entity node
            weight=1.0
        )
Graph Structure:
[Entity Node: DHAKA UNIVERSITY]
   ↑
   |
   | (bipartite edge)
   |
[Knowledge Edge Node: "Dhaka University offers CS programs"]
   ↓
[Entity Node: COMPUTER SCIENCE]
Storage Format (NetworkX → GraphML):
<graphml>
  <node id="DHAKA UNIVERSITY">
    <data key="entity_type">ORGANIZATION</data>
    <data key="description">Premier public university in Bangladesh</data>
    <data key="source_id">chunk-abc123****chunk-def456****chunk-ghi789</data>
    <data key="weight">95</data>
  </node>
  
  <node id="&lt;hyperedge&gt;Dhaka University offers CS programs">
    <data key="hyperedge_name">Dhaka University offers CS programs</data>
    <data key="source_id">chunk-abc123</data>
    <data key="weight">80</data>
  </node>
  
  <edge source="&lt;hyperedge&gt;Dhaka offers CS" target="DHAKA UNIVERSITY" weight="1.0"/>
  <edge source="&lt;hyperedge&gt;Dhaka offers CS" target="COMPUTER SCIENCE" weight="1.0"/>
</graphml>
Phase 5: Vector Embedding (3 Parallel Streams)
# Stream 1: Entity embeddings
entity_data_for_vdb = {
    compute_mdhash_id(entity["entity_name"], prefix="ent-"): {
        "content": entity["entity_name"] + entity["description"],  # Concatenate for richer embedding
        "entity_name": entity["entity_name"],
    }
    for entity in all_entities_data
}
await entity_vdb.upsert(entity_data_for_vdb)

# Stream 2: Hyperedge embeddings
hyperedge_data_for_vdb = {
    compute_mdhash_id(edge["hyperedge_name"], prefix="rel-"): {
        "content": edge["hyperedge_name"],
        "hyperedge_name": edge["hyperedge_name"],
    }
    for edge in all_hyperedges_data
}
await hyperedge_vdb.upsert(hyperedge_data_for_vdb)

# Stream 3: Chunk embeddings (BiG-RAG NEW)
chunk_data_for_vdb = {
    compute_mdhash_id(chunk["content"], prefix="chunk-"): {
        "content": chunk["content"],
        "chunk_id": chunk_id,
        "source_id": chunk["full_doc_id"],
    }
    for chunk_id, chunk in chunks.items()
}
await chunks_vdb.upsert(chunk_data_for_vdb)
Embedding Process:
# graphr1/storage.py lines 82-120 (NanoVectorDB example)
async def upsert(self, data: dict[str, dict]):
    # Extract content for embedding
    contents = [v["content"] for v in data.values()]
    
    # Batch embedding (default: 32 items per batch)
    batches = [contents[i:i+32] for i in range(0, len(contents), 32)]
    
    for batch in batches:
        embeddings = await self.embedding_func(batch)  # Call OpenAI/etc
        
        # Store: {id, embedding, metadata}
        for content, embedding in zip(batch, embeddings):
            self._client.upsert([{
                "__id__": id,
                "__vector__": embedding,
                "content": content,
                "entity_name": metadata["entity_name"],  # Metadata for filtering
                ...
            }])
Why 3 separate vector DBs?
Different embedding strategies:
Entities: Name + description (concept-level)
Edges: Relationship statement (fact-level)
Chunks: Raw text (detail-level)
Different retrieval needs (Path A, B, C)
Industry standard: Separate indices for different content types
Phase 6: Storage Persistence
# graphr1/graphr1.py lines 331-351
await self.full_docs.upsert(new_docs)           # Store original documents
await self.text_chunks.upsert(inserting_chunks) # Store chunks
await self._insert_done()                       # Flush all to disk

async def _insert_done(self):
    tasks = [
        self.full_docs.index_done_callback(),              # Write kv_store_full_docs.json
        self.text_chunks.index_done_callback(),            # Write kv_store_text_chunks.json
        self.llm_response_cache.index_done_callback(),     # Write kv_store_llm_response_cache.json
        self.entities_vdb.index_done_callback(),           # Write vdb_entities.json
        self.hyperedges_vdb.index_done_callback(),         # Write vdb_hyperedges.json
        self.chunks_vdb.index_done_callback(),             # Write vdb_chunks.json
        self.chunk_entity_relation_graph.index_done_callback(),  # Write graph.graphml
    ]
    await asyncio.gather(*tasks)  # Parallel writes
File Structure After Indexing:
working_dir/
├── kv_store_full_docs.json          # Original documents
├── kv_store_text_chunks.json        # All chunks with metadata
├── kv_store_entities.json           # Entity metadata (NOT USED in current code)
├── kv_store_hyperedges.json         # Edge metadata (NOT USED in current code)
├── kv_store_llm_response_cache.json # LLM API cache (for cost savings)
├── vdb_entities.json                # Entity embeddings (NanoVectorDB format)
├── vdb_hyperedges.json              # Edge embeddings
├── vdb_chunks.json                  # Chunk embeddings
└── graph_chunk_entity_relation.graphml  # Bipartite graph structure
📊 DATA FLOW VISUALIZATION
Upload Document "doc.txt"
    ↓
┌────────────────────────────────────────┐
│ Phase 1: Deduplication                 │
│ • MD5 hash → doc-5d41402...            │
│ • Check if exists in corpus            │
│ • Skip if duplicate                    │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Phase 2: Chunking                      │
│ • Split by 1200 tokens, 100 overlap    │
│ • MD5 each chunk → chunk-abc123...     │
│ • Link: chunk → parent doc             │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Phase 3: LLM Entity Extraction         │
│ • Send each chunk to LLM               │
│ • Parse: entities + relationships      │
│ • Iterative gleaning (ask "miss any?") │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Phase 4: Graph Construction            │
│ • Merge duplicate entities             │
│ • Create entity nodes                  │
│ • Create knowledge edge nodes          │
│ • Create bipartite edges               │
│ • Link: entity/edge → source chunks    │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Phase 5: Vector Embedding (Parallel)   │
│ Stream 1: Entity → vdb_entities        │
│ Stream 2: Edge → vdb_hyperedges        │
│ Stream 3: Chunk → vdb_chunks (BiG-RAG) │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Phase 6: Persistence                   │
│ • Full docs → JSON                     │
│ • Chunks → JSON                        │
│ • Graph → GraphML                      │
│ • Vectors → NanoVectorDB/Milvus        │
└────────────────────────────────────────┘
    ↓
Ready for Querying!

For a corpus of 1000 documents (500KB each):
Input: 1000 documents
    ↓
Deduplication: 950 unique documents (50 duplicates removed)
    ↓
Chunking: 950 docs × 50 chunks = 47,500 chunks
    ↓
Entity Extraction: 
    • 150,000 entities extracted
    • 50,000 unique entities (after merging)
    • 80,000 knowledge edges
    ↓
Graph Construction:
    • Nodes: 50,000 entities + 80,000 edges = 130,000 nodes
    • Edges: 300,000 bipartite connections
    ↓
Vector Embeddings:
    • Entity vectors: 50,000 × 1536 dim = 76.8M floats
    • Edge vectors: 80,000 × 1536 dim = 122.8M floats
    • Chunk vectors: 47,500 × 1536 dim = 72.96M floats
    • Total: 272.56M floats ≈ 1GB storage (fp32)
    ↓
Total Storage:
    • Documents: 500MB
    • Graph: 200MB (GraphML)
    • Vectors: 1GB
    • KV stores: 300MB
    • Total: ~2GB
Indexing Time (1000 docs):
Chunking: ~5 min
LLM extraction: ~120 min (with GPT-4, 8 parallel)
Graph construction: ~10 min
Vector embedding: ~15 min
Total: ~150 min (2.5 hours)
Indexing Cost (OpenAI):
LLM extraction: 47,500 chunks × $0.01 = $475
Embeddings: 177,500 items × $0.0001 = $17.75
Total: ~$493
With BiG-RAG optimizations:
LLM cache hits: 50% → $237.50
Local embeddings: $0
Total: ~$238 (50% savings)

⚠️ WEAKNESSES (What Could Be BETTER)
1. No Chunk Vector Search in GraphR1 ⭐⭐⭐ → FIXED in BiG-RAG ⭐⭐⭐⭐⭐
Problem:
# GraphR1 only searches entities + edges
# What if query needs verbatim text from chunks?
Solution (BiG-RAG):
# Add vdb_chunks + Path C retrieval
chunk_results = await vdb_chunks.query(query, top_k=5)
Verdict: ✅ FIXED in BiG-RAG
2. No Semantic Reranking ⭐⭐⭐ → FIXED in BiG-RAG ⭐⭐⭐⭐⭐
Problem:
# Vector search (bi-encoder) may not capture subtle relevance
Solution (BiG-RAG):
# Add cross-encoder reranking
reranked = cross_encoder.rank(query, chunks)
Verdict: ✅ FIXED in BiG-RAG
3. Hardcoded Entity Types ⭐⭐⭐⭐ (Minor Issue)
# graphr1/prompt.py
DEFAULT_ENTITY_TYPES = ["ORGANIZATION", "PERSON", "LOCATION", "EVENT", ...]
Problem:
Domain-specific entities may be missed
User must manually configure entity_types
Solution (Future Enhancement):
# Auto-discover entity types from corpus
entity_types = await llm_discover_entity_types(sample_chunks)
Verdict: ⚠️ Minor issue, configurable via addon_params
4. No Document Metadata ⭐⭐⭐ (Minor Issue)
# Only stores raw content
new_docs = {doc_id: {"content": content}}
Problem:
No title, author, date, URL, etc.
Hard to filter by metadata
Solution (Future Enhancement):
new_docs = {
    doc_id: {
        "content": content,
        "metadata": {
            "title": "Research Paper",
            "author": "John Doe",
            "date": "2025-01-01",
            "source": "arxiv.org/..."
        }
    }
}
Verdict: ⚠️ Minor issue, can be added
5. No Incremental Updates ⭐⭐⭐ (Minor Issue)
Problem:
# If document changes, must re-index entire document
# Can't update just changed paragraphs
Solution (Future Enhancement):
# Delta indexing: Only re-index changed chunks
changed_chunks = diff(old_doc, new_doc)
await reindex(changed_chunks)
Verdict: ⚠️ Complex to implement, acceptable trade-off
6. LLM Entity Extraction Cost ⭐⭐⭐⭐ (Important Issue)
Problem:
# Every chunk → LLM API call → $$$
# 1000 chunks × $0.01 per chunk = $10 per indexing
Solutions:
✅ Already implemented: LLM response cache
Future: Use local LLM (Ollama, LLaMA)
Future: Hybrid NER + LLM (NER for common entities, LLM for relationships)
Verdict: ⚠️ Acceptable with caching, can optimize further
7. No Graph Versioning ⭐⭐ (Minor Issue)
Problem:
# graph.graphml is overwritten
# Can't rollback to previous version
Solution (Future Enhancement):
# Version control for graph
graph_v1.graphml
graph_v2.graphml
Verdict: ⚠️ Nice to have, not critical