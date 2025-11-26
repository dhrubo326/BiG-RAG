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
    normalize_entity_type,  # A2: Entity type validation
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

    # Production KG Pipeline (DEPRECATED - use use_enhanced_pipeline instead)
    use_production_pipeline: bool = False  # DEPRECATED: Use use_enhanced_pipeline
    production_pipeline_config: dict = field(default_factory=lambda: {
        "validation_level": "MODERATE",  # STRICT (99%) | MODERATE (95%) | LENIENT (80%)
        "enable_entity_linking": True,
        "extraction_mode": "semi_structured"  # structured | semi_structured | unstructured
    })

    # Enhanced KG Pipeline (NEW - Phase 1 redesign with extraction strategies)
    use_enhanced_pipeline: bool = False  # Default: False (opt-in for Phase 1)
    enhanced_pipeline_config: dict = field(default_factory=lambda: {
        "validation_level": "MODERATE",  # STRICT (99%) | MODERATE (95%) | LENIENT (80%)
        "enable_entity_linking": True,
        "extraction_strategy": "hybrid",  # NEW: strict | gleaning | hybrid [RECOMMENDED]
        "extraction_mode": "semi_structured"  # structured | semi_structured | unstructured
    })

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        import warnings
        # NEW: Handle deprecated config keys with migration
        if self.use_production_pipeline and not self.use_enhanced_pipeline:
            warnings.warn(
                "'use_production_pipeline' is deprecated. Use 'use_enhanced_pipeline' instead. "
                "Automatically migrating to enhanced pipeline.",
                DeprecationWarning,
                stacklevel=2
            )
            self.use_enhanced_pipeline = True
            # Migrate production config to enhanced config
            if not self.enhanced_pipeline_config:
                self.enhanced_pipeline_config = self.production_pipeline_config.copy()
                # Add extraction_strategy if not present
                if 'extraction_strategy' not in self.enhanced_pipeline_config:
                    self.enhanced_pipeline_config['extraction_strategy'] = 'hybrid'

        # Use centralized logging directory or fallback to working_dir/logs
        from bigrag.config import config
        from pathlib import Path

        # Priority: LOG_DIR env var > centralized logs/bigrag-core > working_dir/logs
        if config.log_dir:
            logs_dir = config.log_dir
        else:
            # Try to find project root by looking for markers
            # Start from current directory and go up
            current = Path.cwd()
            project_root = None

            # Check current directory and up to 4 parent directories
            for _ in range(5):
                # Look for project root markers (not just logs/ directory)
                # Project root should have: bigrag/ package AND logs/ directory
                has_bigrag_package = (current / "bigrag").exists() and (current / "bigrag" / "__init__.py").exists()
                has_logs_dir = (current / "logs").exists() and (current / "logs").is_dir()

                if has_bigrag_package and has_logs_dir:
                    project_root = current
                    break

                if current.parent == current:  # Reached filesystem root
                    break
                current = current.parent

            if project_root:
                # Found project root with both bigrag/ and logs/
                logs_dir = str(project_root / "logs" / "bigrag-core")
            else:
                # Fallback to working_dir/logs for backward compatibility
                logs_dir = os.path.join(self.working_dir, "logs")

        os.makedirs(logs_dir, exist_ok=True)

        log_file = os.path.join(logs_dir, "bigrag.log")
        set_logger(log_file, level=self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        logger.debug(f"Logs directory: {logs_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"BiGRAG init with param:\n  {_print_config}\n")

        # NEW (Phase 1 Step 4): Initialize UnifiedEntityMerger if requested
        self.entity_merger = None
        entity_merge_strategy = self.addon_params.get('entity_merge_strategy', None)
        if entity_merge_strategy:
            try:
                from bigrag.merging.unified_merger import UnifiedEntityMerger
                self.entity_merger = UnifiedEntityMerger(strategy=entity_merge_strategy)
                logger.info(f"[UnifiedMerger] Initialized with strategy={entity_merge_strategy} (standard pipeline)")
            except Exception as e:
                logger.warning(f"[UnifiedMerger] Failed to initialize: {e}. Using default merging.")
                self.entity_merger = None

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
            meta_fields={"entity_id", "entity_name"},  # FIX #1: Store both entity_id and entity_name
            **self.vector_db_storage_cls_kwargs,
        )
        self.vdb_relations = self.vector_db_storage_cls(
            namespace="relations",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"relation_id"},  # FIX #2: Renamed from relation_name for clarity (stores hash ID)
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

    @staticmethod
    def recommend_config(
        sample_documents: list,
        corpus_size: int,
        performance_profile: str = "balanced"
    ) -> dict:
        """
        Recommend optimal pipeline configuration (Phase 1 Step 5).

        Uses pipeline selector to analyze documents and recommend configuration.

        Args:
            sample_documents: Sample of documents (5-10 recommended)
            corpus_size: Total number of documents in corpus
            performance_profile: 'speed', 'balanced', or 'accuracy'

        Returns:
            Dictionary with:
            {
                'pipeline_type': 'standard' or 'enhanced',
                'config': {config_dict},
                'reasoning': [list of reasons],
                'estimated_cost': 'low/medium/high',
                'estimated_time': 'fast/medium/slow',
                'expected_quality': 'good/very_good/excellent'
            }

        Example:
            # Get recommendation
            rec = BiGRAG.recommend_config(
                sample_documents=docs[:10],
                corpus_size=10000,
                performance_profile='speed'
            )

            # Use recommended config (standard pipeline)
            if rec['pipeline_type'] == 'standard':
                rag = BiGRAG(
                    working_dir="./graph",
                    addon_params=rec['config']
                )
            # Use enhanced pipeline
            else:
                from bigrag.enhanced_pipeline import EnhancedKGPipeline
                pipeline = EnhancedKGPipeline(
                    api_key=api_key,
                    **rec['config']
                )
        """
        from bigrag.pipeline_selector import quick_recommend

        recommendation = quick_recommend(
            documents=sample_documents,
            corpus_size=corpus_size,
            performance_profile=performance_profile
        )

        return {
            'pipeline_type': recommendation.pipeline_type.value,
            'config': recommendation.config,
            'reasoning': recommendation.reasoning,
            'estimated_cost': recommendation.estimated_cost,
            'estimated_time': recommendation.estimated_time,
            'expected_quality': recommendation.expected_quality,
            'confidence': recommendation.confidence
        }

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

            # NEW: Enhanced/Production pipeline vs standard pipeline
            if self.use_enhanced_pipeline:
                logger.info(f"[Enhanced Pipeline v1.0] Using extraction strategy: {self.enhanced_pipeline_config.get('extraction_strategy', 'hybrid')}")
                # Process each document with enhanced pipeline
                for doc_key, doc in new_docs.items():
                    await self._process_document_with_enhanced_pipeline(
                        doc_key,
                        doc["content"],
                        doc.get("metadata", {})
                    )
            elif self.use_production_pipeline:
                # DEPRECATED: Still supported but migrated to enhanced
                logger.warning("[Production Pipeline] DEPRECATED - automatically using enhanced pipeline")
                logger.info("[Enhanced Pipeline v1.0] Using extraction strategy: hybrid")
                # Process each document with production pipeline (legacy)
                for doc_key, doc in new_docs.items():
                    await self._process_document_with_production_pipeline(
                        doc_key,
                        doc["content"],
                        doc.get("metadata", {})
                    )
            else:
                # EXISTING: Standard pipeline (unchanged)
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
                    vdb_relations=self.vdb_relations,
                    global_config=asdict(self),
                )
                if maybe_new_kg is None:
                    logger.warning("No new relations and entities found")
                    return
                self.chunk_entity_relation_graph = maybe_new_kg

                await self.full_docs.upsert(new_docs)
                await self.text_chunks.upsert(inserting_chunks)

                # Phase 3.1: Index chunks to vector DB for Path C retrieval (Three-Path Retrieval)
                # This enables direct semantic search on chunks (in addition to entity/edge-based retrieval)
                if self.vdb_chunks is not None:
                    def _build_contextualized_chunk_content(chunk_data: dict) -> str:
                        """Build chunk content with document context prefix for embedding.

                        This enriches chunk embeddings with document metadata (title + category + tags) to make
                        chunks from different documents distinguishable even if content is similar.

                        Example:
                            Input: {"content": "CSE has 180 seats", "doc_title": "RUET", "doc_metadata": {"category": "university", "tags": ["Engineering"]}}
                            Output: "[RUET | university | Engineering] CSE has 180 seats"
                        """
                        content = chunk_data.get("content", "")
                        doc_title = chunk_data.get("doc_title", "")
                        doc_metadata = chunk_data.get("doc_metadata", {})

                        context_parts = []
                        if doc_title:
                            context_parts.append(doc_title)
                        # Add category for better document type distinction
                        if doc_metadata.get("category"):
                            context_parts.append(doc_metadata["category"])
                        if doc_metadata.get("tags"):
                            tags = doc_metadata["tags"]
                            if isinstance(tags, list):
                                context_parts.extend(tags)
                            else:
                                context_parts.append(str(tags))

                        if context_parts:
                            context_prefix = "[" + " | ".join(context_parts) + "] "
                            return context_prefix + content
                        else:
                            return content

                    chunks_for_vdb = {
                        chunk_id: {
                            "content": _build_contextualized_chunk_content(chunk_data),
                            "full_doc_id": chunk_data.get("full_doc_id", ""),
                        }
                        for chunk_id, chunk_data in inserting_chunks.items()
                    }
                    await self.vdb_chunks.upsert(chunks_for_vdb)
                    logger.info(f"[Chunks VDB] Indexed {len(chunks_for_vdb)} chunks for vector search (Path C)")
        finally:
            if update_storage:
                await self._insert_done()

    async def _process_document_with_production_pipeline(
        self,
        doc_id: str,
        content: str,
        metadata: dict,
    ):
        """
        Process a single document through ProductionKGPipeline and insert into BiGRAG storage.

        This method:
        1. Initializes ProductionKGPipeline with API key and config
        2. Processes document through all 5 pipeline phases (chunking, extraction, merging, validation)
        3. Checks validation status (PASS/WARNING/FAIL)
        4. Maps ProductionPipeline chunks to BiGRAG chunk IDs
        5. Stores entities, relations, chunks, and full document
        6. Indexes to all 3 vector DBs (entities, relations, chunks)
        7. Falls back to standard extraction if validation fails or no API key

        Args:
            doc_id: Document ID
            content: Document content (full text, not chunks)
            metadata: Document metadata dict
        """
        from bigrag.production_pipeline import ProductionKGPipeline
        from bigrag.builders.bipartite_graph_builder import build_bipartite_graph_from_pipeline
        from bigrag.utils import compute_mdhash_id

        logger.info(f"[Production Pipeline] Processing document: {doc_id}")

        # Step 1: Get OpenAI API key from .env (REQUIRED for production pipeline)
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            error_msg = (
                "[Production Pipeline] OPENAI_API_KEY not found in environment variables. "
                "Production pipeline requires OpenAI API key. "
                "Please set OPENAI_API_KEY in your .env file or environment."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("[Production Pipeline] API key loaded from OPENAI_API_KEY environment variable")

        # Step 2: Initialize ProductionKGPipeline with config
        try:
            pipeline_config = self.production_pipeline_config
            pipeline = ProductionKGPipeline(
                api_key=api_key,
                model="gpt-4o-mini",  # Default model (can be overridden in config)
                validation_level=pipeline_config.get("validation_level", "MODERATE"),
                extraction_mode=pipeline_config.get("extraction_mode", "semi_structured"),
                enable_entity_linking=pipeline_config.get("enable_entity_linking", True),
            )
            logger.info(f"[Production Pipeline] Initialized with validation={pipeline_config.get('validation_level')}")

        except Exception as e:
            error_msg = f"[Production Pipeline] Failed to initialize pipeline: {e}\nNO FALLBACK: Fix initialization error."
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Step 3: Process document through production pipeline
        try:
            result = await pipeline.process_document(
                markdown_text=content,
                metadata=metadata,
                language="English"  # TODO: Make configurable
            )
            logger.info(f"[Production Pipeline] Extraction complete: {len(result['entities'])} entities, {len(result['relations'])} relations")

        except Exception as e:
            error_msg = f"[Production Pipeline] Processing failed: {e}\nNO FALLBACK: Fix processing error."
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Step 4: Check validation status
        validation = result['validation']
        overall_status = validation['overall_status']

        # Allow WARNING status (with logging), block only on FAIL
        if overall_status == 'WARNING':
            logger.warning(f"[Production Pipeline] Validation WARNING - proceeding with caution")
            logger.warning(f"  Numeric status: {validation['numeric']['status']}")

        if overall_status == 'FAIL':
            numeric_validation = validation['numeric']

            error_details = {
                'overall_status': 'FAIL',
                'numeric_coverage': numeric_validation['numeric_coverage'],
                'numeric_status': numeric_validation['status'],
                'missing_numbers': numeric_validation.get('missing_numbers', []),
                'hallucinated_numbers': numeric_validation.get('hallucinated_numbers', []),
                'validation_level': validation.get('validation_level', 'MODERATE'),
                'recommendations': numeric_validation.get('recommendations', [])
            }

            # Determine threshold based on validation level
            threshold_map = {"STRICT": "95%", "MODERATE": "90%", "LENIENT": "80%"}
            required_threshold = threshold_map.get(error_details['validation_level'], "90%")

            error_message = (
                f"[Production Pipeline] Validation FAILED\n"
                f"  Validation Level: {error_details['validation_level']}\n"
                f"  Overall Status: FAIL\n"
                f"  \n"
                f"  Numeric Validation:\n"
                f"    Status: {error_details['numeric_status']}\n"
                f"    Coverage: {error_details['numeric_coverage']:.2%} (need {required_threshold}+ for {error_details['validation_level']})\n"
                f"    Missing numbers: {len(error_details['missing_numbers'])}\n"
                f"    Hallucinated numbers: {len(error_details['hallucinated_numbers'])}\n"
                f"  \n"
                f"  Recommendations:\n"
            )

            for rec in error_details['recommendations']:
                error_message += f"    - {rec}\n"

            error_message += (
                f"  \n"
                f"  ERROR: Production pipeline validation failed. Processing stopped.\n"
                f"  NO FALLBACK: Fix the validation issues instead of using standard pipeline.\n"
                f"  \n"
                f"  To fix:\n"
                f"  1. Check missing numbers (Bangla numerals not detected?)\n"
                f"  2. Use LENIENT validation level for testing (80%+ threshold)\n"
                f"  3. Or fix the root causes in production pipeline validators\n"
            )

            logger.error(error_message)
            raise ValueError(error_message)

        elif overall_status == 'WARNING':
            logger.warning(
                f"[Production Pipeline] Validation completed with WARNINGS - "
                f"Numeric: {validation['numeric']['numeric_coverage']:.2%}"
            )
        else:  # PASS
            logger.info(
                f"[Production Pipeline] Validation PASSED - "
                f"Numeric: {validation['numeric']['numeric_coverage']:.2%}"
            )

        # Step 5: Build bipartite graph from pipeline results
        try:
            graph_stats = await build_bipartite_graph_from_pipeline(
                pipeline_result=result,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                vdb_entities=self.vdb_entities,
                vdb_relations=self.vdb_relations,
            )
            logger.info(
                f"[Production Pipeline] Graph built: "
                f"{graph_stats.get('entity_nodes', 0)} entities, "
                f"{graph_stats.get('relation_nodes', 0)} relations, "
                f"{graph_stats.get('bipartite_edges', 0)} edges"
            )

        except Exception as e:
            error_msg = f"[Production Pipeline] Graph building failed: {e}\nNO FALLBACK: Fix graph building error."
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Step 6: Store chunks to KV storage (CRITICAL - was missing in original test)
        chunks = result['chunks']
        bigrag_chunks = {}
        production_chunk_to_bigrag_id = {}

        for prod_chunk in chunks:
            # Create BiGRAG chunk ID (hash of content)
            chunk_id = compute_mdhash_id(prod_chunk['content'], prefix='chunk-')

            bigrag_chunks[chunk_id] = {
                "content": prod_chunk['content'],
                "tokens": prod_chunk.get('tokens', 0),  # FIX 2: Changed from [] to 0 (prevents TypeError in aggregation)
                "chunk_order_index": prod_chunk.get('chunk_order_index', 0),
                "full_doc_id": doc_id,
                "doc_title": metadata.get("title", ""),
                "doc_metadata": metadata,
            }

            # Map ProductionPipeline chunk ID -> BiGRAG chunk ID
            prod_chunk_id = prod_chunk.get('chunk_id') or prod_chunk.get('source_id')
            if prod_chunk_id:
                production_chunk_to_bigrag_id[prod_chunk_id] = chunk_id

        await self.text_chunks.upsert(bigrag_chunks)
        logger.info(f"[Production Pipeline] Stored {len(bigrag_chunks)} chunks to KV storage")

        # Step 7: Store full document (CRITICAL - was missing in original test)
        await self.full_docs.upsert({
            doc_id: {
                "content": content,
                "title": metadata.get("title", ""),
                "metadata": metadata
            }
        })
        logger.info(f"[Production Pipeline] Stored full document: {doc_id}")

        # Step 8: Index chunks to vdb_chunks for Path C retrieval (CRITICAL - was missing)
        if self.vdb_chunks is not None:
            doc_title = metadata.get("title", "")
            chunks_for_vdb = {
                chunk_id: {
                    "content": f"[{doc_title}] {chunk_data['content']}" if doc_title else chunk_data['content'],
                    "full_doc_id": doc_id
                }
                for chunk_id, chunk_data in bigrag_chunks.items()
            }
            await self.vdb_chunks.upsert(chunks_for_vdb)
            logger.info(f"[Production Pipeline] Indexed {len(chunks_for_vdb)} chunks to vector DB (Path C)")

        logger.info(f"[Production Pipeline] Document processing complete: {doc_id}")

    async def _process_document_with_enhanced_pipeline(
        self,
        doc_id: str,
        content: str,
        metadata: dict,
    ):
        """
        Process a single document through EnhancedKGPipeline (Phase 1) and insert into BiGRAG storage.

        NEW in Enhanced Pipeline:
        - Extraction strategy configuration (strict/gleaning/hybrid)
        - Version metadata tracking
        - HITL support for failed extractions
        - Preparation for semantic chunking (Step 2)
        - Preparation for gleaning (Step 3)

        This method:
        1. Initializes EnhancedKGPipeline with API key and config
        2. Processes document through all 4 phases (chunking, extraction, merging, validation)
        3. Checks validation status (PASS/WARNING/FAIL)
        4. Maps Enhanced chunks to BiGRAG chunk IDs
        5. Stores entities, relations, chunks, and full document
        6. Indexes to all 3 vector DBs (entities, relations, chunks)

        Args:
            doc_id: Document ID
            content: Document content (full text, not chunks)
            metadata: Document metadata dict
        """
        from bigrag.enhanced_pipeline import EnhancedKGPipeline
        from bigrag.builders.bipartite_graph_builder import build_bipartite_graph_from_pipeline
        from bigrag.utils import compute_mdhash_id

        logger.info(f"[Enhanced Pipeline v1.0] Processing document: {doc_id}")

        # Step 1: Get OpenAI API key from .env
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            error_msg = (
                "[Enhanced Pipeline] OPENAI_API_KEY not found in environment variables. "
                "Enhanced pipeline requires OpenAI API key. "
                "Please set OPENAI_API_KEY in your .env file or environment."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("[Enhanced Pipeline] API key loaded from OPENAI_API_KEY environment variable")

        # Step 2: Initialize EnhancedKGPipeline with config
        try:
            pipeline_config = self.enhanced_pipeline_config
            pipeline = EnhancedKGPipeline(
                api_key=api_key,
                model="gpt-4o-mini",
                validation_level=pipeline_config.get("validation_level", "MODERATE"),
                extraction_mode=pipeline_config.get("extraction_mode", "semi_structured"),
                extraction_strategy=pipeline_config.get("extraction_strategy", "hybrid"),  # NEW
                enable_entity_linking=pipeline_config.get("enable_entity_linking", True),
                dataset_path=self.working_dir  # NEW: For HITL storage
            )
            logger.info(f"[Enhanced Pipeline] Initialized with validation={pipeline_config.get('validation_level')}, strategy={pipeline_config.get('extraction_strategy')}")

        except Exception as e:
            error_msg = f"[Enhanced Pipeline] Failed to initialize pipeline: {e}\nNO FALLBACK: Fix initialization error."
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Step 3: Process document through enhanced pipeline
        try:
            result = await pipeline.process_document(
                markdown_text=content,
                metadata=metadata,
                language="English"  # TODO: Make configurable
            )
            logger.info(f"[Enhanced Pipeline] Extraction complete: {len(result['entities'])} entities, {len(result['relations'])} relations")
            logger.info(f"[Enhanced Pipeline] Pipeline version: {result['pipeline_metadata']['pipeline_version']}")

        except Exception as e:
            error_msg = f"[Enhanced Pipeline] Processing failed: {e}\nNO FALLBACK: Fix processing error."
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Step 4: Check validation status (same as production)
        validation = result['validation']
        overall_status = validation['overall_status']

        if overall_status == 'WARNING':
            logger.warning(f"[Enhanced Pipeline] Validation WARNING - proceeding with caution")
            logger.warning(f"  Numeric status: {validation['numeric']['status']}")

        if overall_status == 'FAIL':
            numeric_validation = validation['numeric']
            error_details = {
                'overall_status': 'FAIL',
                'numeric_coverage': numeric_validation['numeric_coverage'],
                'numeric_status': numeric_validation['status'],
                'missing_numbers': numeric_validation.get('missing_numbers', []),
                'hallucinated_numbers': numeric_validation.get('hallucinated_numbers', []),
                'validation_level': validation.get('validation_level', 'MODERATE'),
                'recommendations': numeric_validation.get('recommendations', [])
            }

            threshold_map = {"STRICT": "95%", "MODERATE": "90%", "LENIENT": "80%"}
            required_threshold = threshold_map.get(error_details['validation_level'], "90%")

            error_msg = (
                f"[Enhanced Pipeline] Document FAILED validation (threshold: {required_threshold}):\n"
                f"  Numeric coverage: {error_details['numeric_coverage']:.2%}\n"
                f"  Status: {error_details['numeric_status']}\n"
                f"  Missing numbers ({len(error_details['missing_numbers'])}): {error_details['missing_numbers'][:10]}\n"
                f"  Hallucinated numbers ({len(error_details['hallucinated_numbers'])}): {error_details['hallucinated_numbers'][:10]}\n"
                f"NO FALLBACK: Document must meet validation threshold."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Step 5: Build bipartite graph from enhanced pipeline result
        try:
            logger.info(f"[Enhanced Pipeline] BEFORE graph building:")
            logger.info(f"  - Entities in result: {len(result.get('entities', []))}")
            logger.info(f"  - Relations in result: {len(result.get('relations', []))}")
            if result.get('entities'):
                logger.info(f"  - First 3 entity names: {[e.get('entity_name', 'NO_NAME') for e in result['entities'][:3]]}")
            if result.get('relations'):
                logger.info(f"  - First 3 relation snippets: {[r.get('content', 'NO_CONTENT')[:50] for r in result['relations'][:3]]}")
            logger.info(f"  - Graph instance type: {type(self.chunk_entity_relation_graph).__name__}")
            logger.info(f"  - Current graph nodes: {self.chunk_entity_relation_graph.graph.number_of_nodes() if hasattr(self.chunk_entity_relation_graph, 'graph') else 'N/A'}")

            graph_stats = await build_bipartite_graph_from_pipeline(
                pipeline_result=result,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                vdb_entities=self.vdb_entities,
                vdb_relations=self.vdb_relations
            )

            logger.info(f"[Enhanced Pipeline] AFTER graph building:")
            logger.info(f"  - Graph stats returned: {graph_stats}")
            logger.info(f"  - Graph nodes after build: {self.chunk_entity_relation_graph.graph.number_of_nodes() if hasattr(self.chunk_entity_relation_graph, 'graph') else 'N/A'}")
            logger.info(f"  - Graph edges after build: {self.chunk_entity_relation_graph.graph.number_of_edges() if hasattr(self.chunk_entity_relation_graph, 'graph') else 'N/A'}")
            logger.info(f"[Enhanced Pipeline] Built bipartite graph for doc {doc_id}")

        except Exception as e:
            error_msg = f"[Enhanced Pipeline] Graph building failed: {e}"
            logger.error(error_msg)
            logger.error(f"  - Exception type: {type(e).__name__}")
            logger.error(f"  - Exception details: {str(e)}")
            import traceback
            logger.error(f"  - Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

        # Step 6: Store chunks to KV storage
        production_chunk_to_bigrag_id = {}
        bigrag_chunks = {}

        for prod_chunk in result['chunks']:
            chunk_id = compute_mdhash_id(prod_chunk['content'], prefix="chunk-")
            bigrag_chunks[chunk_id] = {
                "content": prod_chunk['content'],
                "tokens": prod_chunk.get('tokens', 0),  # FIX 3: Changed from [] to 0 (prevents TypeError in aggregation)
                "chunk_order_index": prod_chunk.get('chunk_order_index', 0),
                "full_doc_id": doc_id,
                "doc_title": metadata.get("title", ""),
                "doc_metadata": metadata,
            }

            prod_chunk_id = prod_chunk.get('chunk_id') or prod_chunk.get('source_id')
            if prod_chunk_id:
                production_chunk_to_bigrag_id[prod_chunk_id] = chunk_id

        await self.text_chunks.upsert(bigrag_chunks)
        logger.info(f"[Enhanced Pipeline] Stored {len(bigrag_chunks)} chunks to KV storage")

        # Step 7: Store full document
        await self.full_docs.upsert({
            doc_id: {
                "content": content,
                "title": metadata.get("title", ""),
                "metadata": metadata
            }
        })
        logger.info(f"[Enhanced Pipeline] Stored full document: {doc_id}")

        # Step 8: Index chunks to vdb_chunks for Path C retrieval
        if self.vdb_chunks is not None:
            doc_title = metadata.get("title", "")
            chunks_for_vdb = {
                chunk_id: {
                    "content": chunk_data["content"],
                    "title": doc_title
                }
                for chunk_id, chunk_data in bigrag_chunks.items()
            }
            await self.vdb_chunks.upsert(chunks_for_vdb)
            logger.info(f"[Enhanced Pipeline] Indexed {len(chunks_for_vdb)} chunks to vector DB (Path C)")

        logger.info(f"[Enhanced Pipeline] Document processing complete: {doc_id}")

    async def _process_document_standard(
        self,
        doc_id: str,
        content: str,
        metadata: dict,
    ):
        """
        Fallback to standard extraction pipeline when production pipeline fails or is unavailable.

        This extracts the existing chunking + extract_entities code path to avoid duplication.
        """
        from bigrag.operate import chunking_by_token_size, extract_entities
        from bigrag.utils import compute_mdhash_id

        logger.info(f"[Standard Pipeline] Processing document: {doc_id}")

        # Chunk document using standard token-based chunking
        chunks = {
            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                **dp,
                "full_doc_id": doc_id,
            }
            for dp in chunking_by_token_size(
                content,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
                tiktoken_model=self.tiktoken_model_name,
                doc_title=metadata.get("title", ""),
                doc_metadata=metadata,
            )
        }

        logger.info(f"[Standard Pipeline] Created {len(chunks)} chunks")

        # Extract entities and relations
        maybe_new_kg = await extract_entities(
            chunks,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            vdb_entities=self.vdb_entities,
            vdb_relations=self.vdb_relations,
            global_config=asdict(self),
        )

        if maybe_new_kg is None:
            logger.warning("[Standard Pipeline] No new relations and entities found")
            return

        self.chunk_entity_relation_graph = maybe_new_kg

        # Store full document and chunks
        await self.full_docs.upsert({doc_id: {"content": content, "title": metadata.get("title", ""), "metadata": metadata}})
        await self.text_chunks.upsert(chunks)

        # Index chunks to vector DB (Path C)
        if self.vdb_chunks is not None:
            def _build_contextualized_chunk_content(chunk_data: dict) -> str:
                content = chunk_data.get("content", "")
                doc_title = chunk_data.get("doc_title", "")
                doc_metadata = chunk_data.get("doc_metadata", {})

                context_parts = []
                if doc_title:
                    context_parts.append(doc_title)
                if doc_metadata.get("category"):
                    context_parts.append(doc_metadata["category"])
                if doc_metadata.get("tags"):
                    tags = doc_metadata["tags"]
                    if isinstance(tags, list):
                        context_parts.extend(tags)
                    else:
                        context_parts.append(str(tags))

                if context_parts:
                    context_prefix = "[" + " | ".join(context_parts) + "] "
                    return context_prefix + content
                else:
                    return content

            chunks_for_vdb = {
                chunk_id: {
                    "content": _build_contextualized_chunk_content(chunk_data),
                    "full_doc_id": chunk_data.get("full_doc_id", ""),
                }
                for chunk_id, chunk_data in chunks.items()
            }
            await self.vdb_chunks.upsert(chunks_for_vdb)
            logger.info(f"[Standard Pipeline] Indexed {len(chunks_for_vdb)} chunks for vector search (Path C)")

        logger.info(f"[Standard Pipeline] Document processing complete: {doc_id}")

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.vdb_entities,
            self.vdb_relations,
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
            # Helper function for contextualized chunk content (same as in ainsert)
            def _build_contextualized_chunk_content(chunk_data: dict) -> str:
                """Build chunk content with document context prefix for embedding."""
                content = chunk_data.get("content", "")
                doc_title = chunk_data.get("doc_title", "")
                doc_metadata = chunk_data.get("doc_metadata", {})

                context_parts = []
                if doc_title:
                    context_parts.append(doc_title)
                if doc_metadata.get("category"):
                    context_parts.append(doc_metadata["category"])
                if doc_metadata.get("tags"):
                    tags = doc_metadata["tags"]
                    if isinstance(tags, list):
                        context_parts.extend(tags)
                    else:
                        context_parts.append(str(tags))

                if context_parts:
                    context_prefix = "[" + " | ".join(context_parts) + "] "
                    return context_prefix + content
                else:
                    return content

            # Insert chunks into vector storage
            all_chunks_data = {}
            all_chunks_data_for_vdb = {}  # Separate dict for VDB with contextualized content
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                # For KV storage: store raw content with metadata
                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                # Preserve metadata if provided
                if chunk_data.get("doc_title"):
                    chunk_entry["doc_title"] = chunk_data["doc_title"]
                if chunk_data.get("doc_metadata"):
                    chunk_entry["doc_metadata"] = chunk_data["doc_metadata"]

                all_chunks_data[chunk_id] = chunk_entry

                # For VDB: store contextualized content for better embedding
                all_chunks_data_for_vdb[chunk_id] = {
                    "content": _build_contextualized_chunk_content(chunk_data),
                    "full_doc_id": chunk_data.get("full_doc_id", ""),
                }

                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.vdb_chunks is not None and all_chunks_data_for_vdb:
                await self.vdb_chunks.upsert(all_chunks_data_for_vdb)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                # A2 Fix: Normalize entity type from user input
                raw_entity_type = entity_data.get("entity_type", "UNKNOWN")
                entity_type = normalize_entity_type(raw_entity_type)
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
                    "entity_type": entity_type,  # Now normalized
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
                        # A2 Fix: Normalize "UNKNOWN" entity type for auto-created nodes
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": normalize_entity_type("UNKNOWN"),  # Will map to "category"
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
                    compute_mdhash_id(dp["entity_name"], prefix="entity-"): {  # UNIFIED: Use "entity-" prefix
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.vdb_entities.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            if self.vdb_relations is not None:
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
                await self.vdb_relations.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query(self, query: str, param: QueryParam = QueryParam(), entity_match=None, relation_match=None):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param, entity_match, relation_match))

    async def aquery(self, query: str, param: QueryParam = QueryParam(), entity_match=None, relation_match=None):
        # All query modes now pass VDB instances directly to kg_query
        # kg_query will handle querying based on param.mode
        # Phase 3.2: Now includes vdb_chunks for Three-Path Retrieval
        response = await kg_query(
            query,
            self.chunk_entity_relation_graph,
            self.vdb_entities,  # Path A: Entity vector DB
            self.vdb_relations,  # Path B: Relation vector DB
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
            await self.vdb_relations.delete_relation(entity_name)
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
            self.vdb_relations,
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
                return {
                    "chunks_deleted": 0,
                    "entities_deleted": 0,
                    "edges_deleted": 0,
                    "document_id": doc_id,
                    "status": "not_found"
                }

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
                # Bug #2 Fix: Use delete() to remove only this document, not drop() which deletes ALL
                await self.full_docs.delete(doc_id)
                logger.info(f"[Document Deletion] Deleted document {doc_id} from full_docs")
                return {
                    "chunks_deleted": 0,
                    "entities_deleted": 0,
                    "edges_deleted": 0,
                    "document_id": doc_id,
                    "status": "no_chunks_found"
                }

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
                            elif role == "relation":
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
                        entity_id = compute_mdhash_id(entity_name, prefix="entity-")  # UNIFIED: Use "entity-" prefix
                        await self.vdb_entities.delete([entity_id])
                except Exception as e:
                    logger.warning(f"Failed to delete entity {entity_name}: {e}")

            # Step 8: Delete orphaned edges from graph and VDB
            for edge_name in edges_to_delete:
                try:
                    await self.chunk_entity_relation_graph.delete_node(edge_name)
                    if self.vdb_relations is not None:
                        # Bug #1 Fix: edge_name is already a hash ID (rel-abc123...)
                        # No need to compute hash again
                        await self.vdb_relations.delete([edge_name])
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

            return {
                "chunks_deleted": deleted_chunks,
                "entities_deleted": len(entities_to_delete),
                "edges_deleted": len(edges_to_delete),
                "document_id": doc_id,
                "status": "success"
            }

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
            self.vdb_relations,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
