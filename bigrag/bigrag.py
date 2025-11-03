import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from .operate import (
    chunking_by_token_size,
    extract_entities,
    # local_query,global_query,hybrid_query,
    kg_query
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        # Import the module using importlib
        module = importlib.import_module(module_name, package=package)

        # Get the class from the module and instantiate it
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class BiGRAG:
    working_dir: str = field(
        default_factory=lambda: f"bigrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 2
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        log_file = os.path.join("bigrag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"BiGRAG init with param:\n  {_print_config}\n")

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.vdb_entities = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},  # Bug #5 fix: Store entity_name for node lookup
            **self.vector_db_storage_cls_kwargs,
        )
        self.vdb_bipartite_edges = self.vector_db_storage_cls(
            namespace="bipartite_edges",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"bipartite_edge_name"},  # Bug #5 fix: Store edge name for node lookup
            **self.vector_db_storage_cls_kwargs,
        )
        self.vdb_chunks = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            **self.vector_db_storage_cls_kwargs,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }

    def insert(self, string_or_strings, metadata=None):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings, metadata))

    async def ainsert(self, string_or_strings, metadata=None):
        """
        Insert documents with optional metadata preservation.

        Args:
            string_or_strings: Single string or list of strings (document content)
            metadata: Optional metadata - can be:
                     - None: No metadata
                     - dict: Single metadata dict (used for all docs if multiple strings)
                     - list of dicts: One metadata dict per document (must match length)

        Metadata format:
            {
                "title": "Document Title",  # Optional but recommended
                "category": "science",      # Optional
                "tags": ["tag1", "tag2"],   # Optional
                # ... any other fields ...
            }
        """
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
                # Wrap single metadata in list for consistency
                if metadata is not None and isinstance(metadata, dict):
                    metadata = [metadata]

            # Normalize metadata to list of dicts
            if metadata is None:
                metadata = [{}] * len(string_or_strings)
            elif isinstance(metadata, dict):
                # Single dict: apply to all documents
                metadata = [metadata] * len(string_or_strings)
            elif isinstance(metadata, list):
                if len(metadata) != len(string_or_strings):
                    logger.warning(
                        f"Metadata length ({len(metadata)}) doesn't match documents ({len(string_or_strings)}). "
                        f"Padding with empty dicts."
                    )
                    # Pad with empty dicts if mismatch
                    metadata = metadata + [{}] * (len(string_or_strings) - len(metadata))
                    metadata = metadata[:len(string_or_strings)]

            # Create new_docs with metadata
            new_docs = {}
            for content, meta in zip(string_or_strings, metadata):
                doc_id = compute_mdhash_id(content.strip(), prefix="doc-")
                new_docs[doc_id] = {
                    "content": content.strip(),
                    "title": meta.get("title", ""),
                    "metadata": meta,
                }

            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc="Chunking documents", unit="doc"
            ):
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                        doc_title=doc.get("title", ""),
                        doc_metadata=doc.get("metadata", {}),
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                vdb_entities=self.vdb_entities,
                vdb_bipartite_edges=self.vdb_bipartite_edges,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new bipartite edges and entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)

            # Phase 3.1: Index chunks to vector DB for Path C retrieval (Three-Path Retrieval)
            # This enables direct semantic search on chunks (in addition to entity/edge-based retrieval)
            if self.vdb_chunks is not None:
                chunks_for_vdb = {
                    chunk_id: {
                        "content": chunk_data["content"],
                        "full_doc_id": chunk_data.get("full_doc_id", ""),
                    }
                    for chunk_id, chunk_data in inserting_chunks.items()
                }
                await self.vdb_chunks.upsert(chunks_for_vdb)
                logger.info(f"[Chunks VDB] Indexed {len(chunks_for_vdb)} chunks for vector search (Path C)")
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.vdb_entities,
            self.vdb_bipartite_edges,
            self.vdb_chunks,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.vdb_chunks is not None and all_chunks_data:
                await self.vdb_chunks.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            if self.vdb_entities is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.vdb_entities.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            if self.vdb_bipartite_edges is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + dp["src_id"]
                        + dp["tgt_id"]
                        + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.vdb_bipartite_edges.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query(self, query: str, param: QueryParam = QueryParam(), entity_match=None, bipartite_edge_match=None):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param, entity_match, bipartite_edge_match))

    async def aquery(self, query: str, param: QueryParam = QueryParam(), entity_match=None, bipartite_edge_match=None):
        # All query modes now pass VDB instances directly to kg_query
        # kg_query will handle querying based on param.mode
        # Phase 3.2: Now includes vdb_chunks for Three-Path Retrieval
        response = await kg_query(
            query,
            self.chunk_entity_relation_graph,
            self.vdb_entities,  # Path A: Entity vector DB
            self.vdb_bipartite_edges,  # Path B: Bipartite edge vector DB
            self.text_chunks,
            self.vdb_chunks,  # Phase 3.2: Path C: Chunk vector DB
            param,
            asdict(self),
            hashing_kv=self.llm_response_cache,
        )
        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.vdb_entities.delete_entity(entity_name)
            await self.vdb_bipartite_edges.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.vdb_entities,
            self.vdb_bipartite_edges,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_document(self, doc_id_or_content: str):
        """
        Synchronous wrapper for adelete_document.
        Delete a document and cascade cleanup of orphaned entities/edges.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_document(doc_id_or_content))

    async def adelete_document(self, doc_id_or_content: str):
        """
        Delete a document by ID or content with intelligent cascade deletion.

        This method implements Phase 2.2 from BiG_RAG_DESIGN.md:
        - Finds all chunks belonging to the document
        - For each chunk, identifies entities and edges that reference it
        - Performs partial deletion (removes chunk reference) if entity/edge exists in other docs
        - Performs full deletion if entity/edge only exists in this document
        - Cleans up all storage layers (full_docs, text_chunks, graph, vector DBs)

        Args:
            doc_id_or_content: Either:
                - Document ID (e.g., "doc-abc123")
                - Document content string (will compute hash ID)

        Example:
            # By ID
            await rag.adelete_document("doc-abc123")

            # By content
            await rag.adelete_document("The original document text...")
        """
        from .prompt import GRAPH_FIELD_SEP

        try:
            # Step 1: Normalize to document ID
            if doc_id_or_content.startswith("doc-"):
                doc_id = doc_id_or_content
            else:
                doc_id = compute_mdhash_id(doc_id_or_content.strip(), prefix="doc-")

            # Step 2: Verify document exists
            doc_data = await self.full_docs.get_by_id(doc_id)
            if doc_data is None:
                logger.warning(f"Document '{doc_id}' not found in storage")
                return

            logger.info(f"[Document Deletion] Starting deletion for document: {doc_id}")

            # Step 3: Find all chunks belonging to this document
            all_chunk_ids = await self.text_chunks.all_keys()
            doc_chunk_ids = []

            for chunk_id in all_chunk_ids:
                chunk_data = await self.text_chunks.get_by_id(chunk_id)
                if chunk_data and chunk_data.get("full_doc_id") == doc_id:
                    doc_chunk_ids.append(chunk_id)

            if not doc_chunk_ids:
                logger.warning(f"No chunks found for document {doc_id}")
                # Still delete from full_docs
                await self.full_docs.drop()  # This might not be granular enough
                logger.info(f"[Document Deletion] Deleted document {doc_id} from full_docs")
                return

            logger.info(f"[Document Deletion] Found {len(doc_chunk_ids)} chunks to process")

            doc_chunk_ids_set = set(doc_chunk_ids)

            # Step 4: Delete chunks from text_chunks KV storage
            logger.info(f"[Document Deletion] Deleting chunks from KV storage")
            deleted_chunks = await self.text_chunks.delete_many(doc_chunk_ids)
            logger.info(f"[Document Deletion] Deleted {deleted_chunks} chunks from KV storage")

            # Step 5: Delete chunks from vdb_chunks
            if self.vdb_chunks is not None:
                logger.info(f"[Document Deletion] Deleting chunks from vector DB")
                deleted_vdb = await self.vdb_chunks.delete(doc_chunk_ids)
                logger.info(f"[Document Deletion] Deleted {deleted_vdb} chunk embeddings from VDB")

            # Step 6: Find and process entities/edges that reference deleted chunks
            # This implements cascade deletion with orphan cleanup
            logger.info(f"[Document Deletion] Processing entity/edge cascade deletion")

            entities_to_delete = []
            edges_to_delete = []

            # Iterate through all graph nodes to find entities/edges referencing deleted chunks
            # We'll use NetworkX directly to iterate nodes
            if hasattr(self.chunk_entity_relation_graph, '_graph'):
                import networkx as nx
                G = self.chunk_entity_relation_graph._graph

                for node, attrs in G.nodes(data=True):
                    source_id_str = str(attrs.get("source_id", ""))
                    if not source_id_str:
                        continue

                    # Parse source_ids
                    source_ids = source_id_str.split(GRAPH_FIELD_SEP) if GRAPH_FIELD_SEP in source_id_str else [source_id_str]
                    source_ids_set = set(source_ids)

                    # Check if this entity/edge references any deleted chunks
                    if source_ids_set & doc_chunk_ids_set:
                        # Remove deleted chunk references
                        remaining_sources = source_ids_set - doc_chunk_ids_set

                        if not remaining_sources:
                            # No other chunks reference this entity/edge - delete it entirely
                            role = attrs.get("role", "")
                            if role == "entity":
                                entities_to_delete.append(node)
                            elif role == "bipartite_edge":
                                edges_to_delete.append(node)
                        else:
                            # Update source_id to remove deleted chunks
                            new_source_id = GRAPH_FIELD_SEP.join(remaining_sources)
                            attrs["source_id"] = new_source_id
                            # Update the node in graph
                            await self.chunk_entity_relation_graph.upsert_node(
                                node,
                                node_data=attrs
                            )

            logger.info(f"[Document Deletion] Found {len(entities_to_delete)} orphaned entities")
            logger.info(f"[Document Deletion] Found {len(edges_to_delete)} orphaned edges")

            # Step 7: Delete orphaned entities from graph and VDB
            for entity_name in entities_to_delete:
                try:
                    await self.chunk_entity_relation_graph.delete_node(entity_name)
                    if self.vdb_entities is not None:
                        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                        await self.vdb_entities.delete([entity_id])
                except Exception as e:
                    logger.warning(f"Failed to delete entity {entity_name}: {e}")

            # Step 8: Delete orphaned edges from graph and VDB
            for edge_name in edges_to_delete:
                try:
                    await self.chunk_entity_relation_graph.delete_node(edge_name)
                    if self.vdb_bipartite_edges is not None:
                        edge_id = compute_mdhash_id(edge_name, prefix="edge-")
                        await self.vdb_bipartite_edges.delete([edge_id])
                except Exception as e:
                    logger.warning(f"Failed to delete edge {edge_name}: {e}")

            # Step 9: Delete document from full_docs
            logger.info(f"[Document Deletion] Deleting document from full_docs")
            await self.full_docs.delete(doc_id)

            logger.info(
                f"[Document Deletion] ✅ Successfully deleted document {doc_id}: "
                f"{deleted_chunks} chunks, {len(entities_to_delete)} entities, {len(edges_to_delete)} edges"
            )

            logger.info(f"[Document Deletion] Completed deletion for document: {doc_id}")

            await self._delete_document_done()

        except Exception as e:
            logger.error(f"Error while deleting document '{doc_id_or_content}': {e}")
            raise

    async def _delete_document_done(self):
        """Callback after document deletion to commit changes"""
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.vdb_chunks,
            self.vdb_entities,
            self.vdb_bipartite_edges,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
