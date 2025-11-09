import asyncio
import json
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .constants import DEFAULT_ENTITY_TYPES, DEFAULT_LLM_CONCURRENCY


# ========================
# Entity Type Validation (A2)
# ========================

TYPE_NORMALIZATION_MAP = {
    # Teams & Organizations
    "TEAM": "organization",
    "CLUB": "organization",
    "LEAGUE": "organization",
    "ORGANIZATION": "organization",
    "ORG": "organization",
    "COMPANY": "organization",
    "CORPORATION": "organization",

    # People
    "PLAYER": "person",
    "PERSON": "person",
    "PEOPLE": "person",
    "INDIVIDUAL": "person",
    "ATHLETE": "person",
    "COACH": "person",
    "MANAGER": "person",

    # Places
    "LOCATION": "geo",
    "GEO": "geo",
    "PLACE": "geo",
    "CITY": "geo",
    "COUNTRY": "geo",
    "REGION": "geo",
    "VENUE": "geo",
    "STADIUM": "geo",

    # Events
    "EVENT": "event",
    "TOURNAMENT": "event",
    "CHAMPIONSHIP": "event",
    "MATCH": "event",
    "GAME": "event",
    "COMPETITION": "event",
    "SEASON": "event",

    # Abstract/Other
    "CONCEPT": "category",
    "CATEGORY": "category",
    "STATISTIC": "category",
    "METRIC": "category",
    "OBJECT": "category",
    "THING": "category",
    "OTHER": "category",
    "TIME": "category",
    "DATE": "category",
}


def normalize_entity_type(extracted_type: str, allowed_types: list = None) -> str:
    """
    Normalize entity type from LLM extraction to allowed types.

    This handles cases where LLM extracts types like "TEAM", "STATISTIC", etc.
    that are not in the configured allowed types list.

    Args:
        extracted_type: Raw entity type from LLM extraction
        allowed_types: List of allowed types (default: DEFAULT_ENTITY_TYPES)

    Returns:
        Normalized lowercase entity type

    Examples:
        normalize_entity_type("TEAM") -> "organization"
        normalize_entity_type("person") -> "person"
        normalize_entity_type("UNKNOWN") -> "category" (with warning)
    """
    if allowed_types is None:
        allowed_types = DEFAULT_ENTITY_TYPES

    # Strip quotes and normalize to uppercase for mapping lookup
    # LLM sometimes outputs types with quotes: "person" or 'person'
    normalized_upper = extracted_type.strip().strip('"').strip("'").strip().upper()

    # Check if it's in the normalization map
    if normalized_upper in TYPE_NORMALIZATION_MAP:
        return TYPE_NORMALIZATION_MAP[normalized_upper]

    # Check if it's already a valid type (case-insensitive)
    for allowed in allowed_types:
        if normalized_upper == allowed.upper():
            return allowed.lower()

    # Unknown type - log warning and fallback to category
    logger.warning(f"Unknown entity type '{extracted_type}' - using fallback 'category'")
    return "category"


def chunking_by_token_size(
    content: str,
    overlap_token_size=128,
    max_token_size=1024,
    tiktoken_model="gpt-4o",
    doc_title: str = "",
    doc_metadata: dict = None,
):
    """
    Chunk content by token size with optional metadata preservation.

    Args:
        content: Text content to chunk
        overlap_token_size: Token overlap between chunks
        max_token_size: Maximum tokens per chunk
        tiktoken_model: Tokenizer model name
        doc_title: Document title for context (optional)
        doc_metadata: Additional metadata dict (optional)

    Returns:
        List of chunk dicts with tokens, content, chunk_order_index, doc_title, doc_metadata
    """
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        chunk = {
            "tokens": min(max_token_size, len(tokens) - start),
            "content": chunk_content.strip(),
            "chunk_order_index": index,
        }
        # Preserve metadata if provided
        if doc_title:
            chunk["doc_title"] = doc_title
        if doc_metadata:
            chunk["doc_metadata"] = doc_metadata
        results.append(chunk)
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """
    Summarize entity/relation descriptions when they exceed token limits.

    Note (B5): This function doesn't use semaphore control because:
    1. Summarization is rare (only when descriptions > summary_max_tokens)
    2. Already indirectly rate-limited by merge function concurrency
    3. Main rate limit risk is in extract_entities (addressed with semaphore)
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    # Normalize entity type to ensure consistency (A2)
    raw_entity_type = clean_str(record_attributes[2])
    entity_type = normalize_entity_type(raw_entity_type)
    entity_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 50.0
    )
    hyper_relation = now_hyper_relation
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        weight=weight,
        hyper_relation=hyper_relation,
        source_id=entity_source_id,
    )


async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"bipartite_edge"':
        return None
    # add this record as edge
    knowledge_fragment = clean_str(record_attributes[1])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )

    # A1: Generate hash-based ID instead of using content directly
    # This reduces GraphML file size by 30-40% and improves query performance
    from .constants import BIPARTITE_EDGE_PREFIX
    edge_id = compute_mdhash_id(knowledge_fragment, prefix=BIPARTITE_EDGE_PREFIX)

    return dict(
        hyper_relation=edge_id,  # Hash ID: "rel-abc123..."
        hyper_relation_content=knowledge_fragment,  # Store content separately
        weight=weight,
        source_id=edge_source_id,
    )
    

async def _merge_bipartite_edges_then_upsert(
    bipartite_edge_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Merge and upsert bipartite edge nodes with weight aggregation.

    A1: Now accepts hash-based IDs and stores content as node attribute.

    Weight Semantics (A3):
    - weight = sum of completeness scores across all occurrences
    - Range: 0 to N×10 (where N = number of chunks mentioning this relation)
    - Higher weight = more frequently mentioned + higher completeness
    - No normalization (intentional - preserves frequency signal)

    Args:
        bipartite_edge_name: Hash ID of the edge (e.g., "rel-abc123...")
        nodes_data: List of dicts with hyper_relation_content, weight, source_id
        knowledge_graph_inst: Graph storage instance
        global_config: Configuration dict

    Returns:
        Node data dict with bipartite_edge_name
    """
    already_weights = []
    already_source_ids = []

    already_bipartite_edge = await knowledge_graph_inst.get_node(bipartite_edge_name)
    if already_bipartite_edge is not None:
        already_weights.append(already_bipartite_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_bipartite_edge["source_id"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    # A1: Extract content from first occurrence (all should be same since same hash)
    # Content is now stored as node attribute instead of being the ID
    content = nodes_data[0].get("hyper_relation_content", "") if nodes_data else ""

    node_data = dict(
        role="bipartite_edge",
        content=content,  # A1: Store content as attribute
        weight=weight,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        bipartite_edge_name,
        node_data=node_data,
    )
    node_data["bipartite_edge_name"] = bipartite_edge_name
    node_data["bipartite_edge_content"] = content  # For VDB upsertion
    return node_data


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Merge and upsert entity nodes with weight aggregation.

    Weight Semantics (A3):
    - weight = sum of importance scores (key_score) across all occurrences
    - Range: 0 to N×100 (where N = number of chunks mentioning this entity)
    - Higher weight = more frequently mentioned + higher LLM importance scores
    - Used for ranking entities by significance in the knowledge graph
    - No normalization (intentional - preserves frequency signal)

    Examples:
      - 400+: Very central entity (4+ mentions, high scores)
      - 200-399: Important entity (2-3 mentions)
      - 100-199: Mentioned entity (1-2 mentions)
      - 50-99: Peripheral entity (1 mention, low score)

    Args:
        entity_name: Name of the entity (uppercase, e.g., "LIONEL MESSI")
        nodes_data: List of dicts with entity_type, description, weight, source_id
        knowledge_graph_inst: Graph storage instance
        global_config: Configuration dict

    Returns:
        Node data dict with entity_name
    """
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_weights = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
        # Preserve existing weight if present
        if "weight" in already_node:
            already_weights.append(already_node["weight"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    # Aggregate weights from all occurrences (same as bipartite edges)
    weight = sum([dp.get("weight", 0) for dp in nodes_data] + already_weights)

    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        role="entity",
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        weight=weight,  # Bug fix: include weight in node data
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_data = []
    
    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]
        
        already_weights = []
        already_source_ids = []
        
        if await knowledge_graph_inst.has_edge(hyper_relation, entity_name):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_name)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
        
        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(
            set([source_id] + already_source_ids)
        )

        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_name,
            edge_data=dict(
                weight=weight,
                source_id=source_id,
            ),
        )

        edge_data.append(dict(
            src_id=hyper_relation,
            tgt_id=entity_name,
            weight=weight,
        ))

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    vdb_entities: BaseVectorStorage,
    vdb_bipartite_edges: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    # B5: Create semaphore to limit concurrent LLM API calls (prevents rate limits)
    llm_concurrency = global_config.get("llm_concurrency", DEFAULT_LLM_CONCURRENCY)
    llm_semaphore = asyncio.Semaphore(llm_concurrency)

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),  # Fixed: Uncommented for prompt compatibility
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        # Extract metadata for context enhancement (Phase 2.1 improvement)
        doc_title = chunk_dp.get("doc_title", "")
        doc_metadata = chunk_dp.get("doc_metadata", {})

        # Build context-enriched input text
        context_parts = []
        if doc_title:
            context_parts.append(f"Document Title: {doc_title}")
        if doc_metadata:
            # Format metadata nicely
            metadata_str = ", ".join(
                f"{k}: {v}" for k, v in doc_metadata.items()
                if k != "title" and v  # Skip empty values and title (already shown)
            )
            if metadata_str:
                context_parts.append(f"Document Context: {metadata_str}")

        # Combine context with content
        if context_parts:
            enriched_content = "\n".join(context_parts) + "\n\n" + content
        else:
            enriched_content = content

        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=enriched_content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=enriched_content)

        # B5: Use semaphore to limit concurrent LLM calls
        async with llm_semaphore:
            final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            async with llm_semaphore:
                glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            async with llm_semaphore:
                if_loop_result: str = await use_llm_func(
                    if_loop_prompt, history_messages=history
                )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation=""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_relation = await _handle_single_hyperrelation_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[if_relation["hyper_relation"]].append(
                    if_relation
                )
                now_hyper_relation = if_relation["hyper_relation"]
                
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, now_hyper_relation
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
            
    logger.info("Inserting bipartite edges into storage...")
    all_bipartite_edges_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_bipartite_edges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting bipartite edges",
        unit="entity",
    ):
        all_bipartite_edges_data.append(await result)
            
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_bipartite_edges_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any bipartite edges and entities, maybe your LLM is not working"
        )
        return None

    if not len(all_bipartite_edges_data):
        logger.warning("Didn't extract any bipartite edges")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    # B3: Wrap VDB operations with retry mechanism
    from .utils import safe_operation_with_retry
    from .constants import DEFAULT_MAX_RETRIES

    if vdb_bipartite_edges is not None:
        # A1: bipartite_edge_name is already a hash ID (rel-abc123...)
        # No need to hash again. Use bipartite_edge_content for vector embedding.
        data_for_vdb = {
            dp["bipartite_edge_name"]: {  # Already hash ID
                "content": dp.get("bipartite_edge_content", ""),  # Use actual content for embedding
                "bipartite_edge_name": dp["bipartite_edge_name"],
            }
            for dp in all_bipartite_edges_data
        }
        await safe_operation_with_retry(
            lambda: vdb_bipartite_edges.upsert(data_for_vdb),
            "VDB upsert bipartite edges",
            context=f"{len(data_for_vdb)} edges",
            max_retries=global_config.get("api_retry_attempts", DEFAULT_MAX_RETRIES),
        )

    if vdb_entities is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await safe_operation_with_retry(
            lambda: vdb_entities.upsert(data_for_vdb),
            "VDB upsert entities",
            context=f"{len(data_for_vdb)} entities",
            max_retries=global_config.get("api_retry_attempts", DEFAULT_MAX_RETRIES),
        )

    return knowledge_graph_inst


def _format_knowledge_as_string(knowledge_list: list[dict]) -> str:
    """
    Convert structured knowledge list to clean string for LLM context injection.

    Args:
        knowledge_list: List of dicts with structure:
            {
                "<knowledge>": "text content",
                "<coherence>": 0.95,
                "<source_ids>": ["id1", "id2"],
                "<type>": "entity" | "relation" | "chunk"
            }

    Returns:
        Clean, newline-separated knowledge text without metadata.
        Already sorted by coherence (highest first) from _build_query_context.

    Example:
        Input: [
            {"<knowledge>": "Messi plays for Inter Miami", "<coherence>": 0.95},
            {"<knowledge>": "Messi won World Cup 2022", "<coherence>": 0.88}
        ]
        Output: "Messi plays for Inter Miami\n\nMessi won World Cup 2022"
    """
    if not knowledge_list:
        return ""

    # Extract knowledge text only (no metadata clutter for LLM)
    knowledge_texts = [
        item.get("<knowledge>", "").strip()
        for item in knowledge_list
        if item.get("<knowledge>")
    ]

    # Join with double newlines for readability
    return "\n\n".join(knowledge_texts)


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    vdb_entities: BaseVectorStorage,  # Bug #6 Fix: Correct type annotation
    vdb_bipartite_edges: BaseVectorStorage,  # Bug #6 Fix: Correct type annotation
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    vdb_chunks: BaseVectorStorage,  # Phase 3.2: Added vdb_chunks parameter
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:

    hl_keywords = query
    ll_keywords = query
    keywords = [ll_keywords, hl_keywords]
    knowledge_list = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        vdb_entities,
        vdb_bipartite_edges,
        text_chunks_db,
        vdb_chunks,  # Phase 3.2: Pass vdb_chunks
        query_param,
    )

    # Bug #1 Fix: Format structured knowledge as string for LLM context
    return _format_knowledge_as_string(knowledge_list)



async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    vdb_entities: BaseVectorStorage,
    vdb_bipartite_edges: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    vdb_chunks: BaseVectorStorage,  # Phase 3.2: Added vdb_chunks parameter
    query_param: QueryParam,
):
    """
    Three-Path Retrieval Context Builder (Phase 3.2 Enhancement)

    Combines three retrieval paths:
    - Path A: Entity-based (structural, high-level)
    - Path B: Bipartite edge-based (relational, knowledge fragments)
    - Path C: Chunk-based (semantic, raw text)

    Returns 5 structured knowledge items + 5 chunk items = 10 total context items
    """

    ll_kewwords, hl_keywrds = query[0], query[1]

    # Path A: Entity retrieval
    knowledge_list_1 = await _get_node_data(
        ll_kewwords,
        knowledge_graph_inst,
        vdb_entities,
        text_chunks_db,
        query_param,
    )

    # Path B: Bipartite edge retrieval
    knowledge_list_2 = await _get_edge_data(
        hl_keywrds,
        knowledge_graph_inst,
        vdb_bipartite_edges,
        text_chunks_db,
        query_param,
    )

    # Phase 3.2: Collect source IDs from Path A + B for Path C
    entity_source_ids = set()
    for _, source_ids in knowledge_list_1:
        if source_ids:
            entity_source_ids.update(source_ids if isinstance(source_ids, (list, set)) else [source_ids])

    edge_source_ids = set()
    for _, source_ids in knowledge_list_2:
        if source_ids:
            edge_source_ids.update(source_ids if isinstance(source_ids, (list, set)) else [source_ids])

    # Path C: Chunk retrieval (direct + indirect from Path A + B)
    knowledge_list_3 = await _get_chunk_data(
        ll_kewwords,  # Use same query as entity search
        vdb_chunks,
        text_chunks_db,
        entity_source_ids,
        edge_source_ids,
        query_param,
    )

    # Build knowledge with scores from all three retrieval paths using RRF
    # Also track source IDs for evaluation purposes
    know_score = dict()
    know_sources = dict()  # Track source IDs for each knowledge item
    know_type = dict()     # Track which path contributed this knowledge

    # Path A contributions
    for i, (k, source_ids) in enumerate(knowledge_list_1):
        if k not in know_score:
            know_score[k] = 0
            know_sources[k] = set()
            know_type[k] = "entity"
        score = 1/(i+1)
        know_score[k] += score
        if source_ids:
            know_sources[k].update(source_ids if isinstance(source_ids, (list, set)) else [source_ids])

    # Path B contributions
    for i, (k, source_ids) in enumerate(knowledge_list_2):
        if k not in know_score:
            know_score[k] = 0
            know_sources[k] = set()
            know_type[k] = "bipartite_edge"
        score = 1/(i+1)
        know_score[k] += score
        if source_ids:
            know_sources[k].update(source_ids if isinstance(source_ids, (list, set)) else [source_ids])

    # Path C contributions with weighted RRF scoring (Modified Approach 2)
    chunk_knowledge = []

    # Sort chunk candidates by their original scores (cosine for direct, 0.5 for indirect)
    # This ensures better chunks appear first regardless of source
    sorted_candidates = sorted(knowledge_list_3,
                              key=lambda x: x.get("score", 0),
                              reverse=True)

    # Apply weighted RRF based on source type
    for i, chunk in enumerate(sorted_candidates):
        if chunk["source"] == "direct_vector":
            # Direct chunks get full RRF weight (semantic relevance)
            score = 1.0 * (1/(i+1))
        else:  # indirect_graph
            # Indirect chunks get 70% weight (structural relevance)
            score = 0.7 * (1/(i+1))

        chunk_knowledge.append({
            "content": chunk["content"],
            "score": score,
            "sources": [chunk["source_id"]],
            "type": chunk["source"]  # Preserve source type for debugging
        })

    # Phase 3.4: Apply semantic reranking to chunks if enabled
    if query_param.enable_reranking and chunk_knowledge:
        try:
            from .reranker import rerank_chunks
            # Prepare chunks for reranking: (content, source_ids)
            chunk_candidates = [(c["content"], c["sources"]) for c in chunk_knowledge]
            # Rerank and get top-5
            reranked = await rerank_chunks(
                query=ll_kewwords,  # Use original query
                chunks=chunk_candidates,
                top_k=5,
                use_reranking=True
            )
            # Update chunk_knowledge with reranked results and scores
            chunk_knowledge = [
                {
                    "content": content,
                    "score": score,
                    "sources": sources,
                    "type": "chunk_reranked"
                }
                for content, sources, score in reranked
            ]
            logger.info("[Reranking] Applied cross-encoder reranking to chunks")
        except Exception as e:
            logger.warning(f"[Reranking] Failed, using original ranking: {e}")
            # Keep original chunk_knowledge

    # Take top-5 structured knowledge (from Path A + B)
    structured_knowledge = sorted(know_score.items(), key=lambda x: x[1], reverse=True)[:5]

    # Combine: 5 structured + 5 chunks = 10 total
    knowledge = []

    # Add structured knowledge (entities + bipartite edges)
    for k, score in structured_knowledge:
        sources = list(know_sources.get(k, []))
        knowledge.append({
            "<knowledge>": k,
            "<coherence>": round(score, 3),
            "<source_ids>": sources,
            "<type>": know_type.get(k, "unknown")  # Add type for debugging
        })

    # Add chunk knowledge (already top-5 after reranking)
    for chunk in chunk_knowledge[:5]:  # Top-5 chunks
        knowledge.append({
            "<knowledge>": chunk["content"],
            "<coherence>": round(chunk["score"], 3),
            "<source_ids>": chunk["sources"],
            "<type>": chunk["type"]
        })

    logger.info(f"[Three-Path Retrieval] Returning {len(knowledge)} items: "
                f"{len(structured_knowledge)} structured + {len(chunk_knowledge[:5])} chunks")

    return knowledge


async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    vdb_entities: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # Fixed: Actually query the vector database instead of assigning the object
    results = await vdb_entities.query(query, top_k=query_param.top_k)
    if not results or not len(results):  # Check for None or empty
        return []  # Return empty list when no results (not empty strings)
    # Bug #4 Fix: Use defensive dict access to prevent KeyError
    # Extract entity names from query results (Bug #5 fix: use entity_name, not hash ID)
    results = [r.get("entity_name") for r in results if "entity_name" in r]
    # get entity information
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    # get entity degree
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    # Extract knowledge and source IDs together for evaluation
    knowledge_list = []
    for s in use_relations:
        # A1 Fix: Description now contains actual content from _find_most_related_edges_from_entities
        description = s["description"]
        # Extract source_ids from the relation (chunks where this relation appears)
        source_ids = []
        if "source_id" in s and s["source_id"]:
            # source_id may contain multiple IDs separated by GRAPH_FIELD_SEP
            source_ids = s["source_id"].split(GRAPH_FIELD_SEP) if isinstance(s["source_id"], str) else [s["source_id"]]
        knowledge_list.append((description, source_ids))
    return knowledge_list


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(e)
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )

    # A1 Fix: Fetch node data for bipartite edge nodes to get content
    # k[1] could be entity or bipartite edge node - need to check which
    bipartite_edge_ids = [k[1] for k in all_edges if k[1].startswith("rel-")]
    bipartite_node_data = {}
    if bipartite_edge_ids:
        nodes = await asyncio.gather(
            *[knowledge_graph_inst.get_node(node_id) for node_id in bipartite_edge_ids]
        )
        bipartite_node_data = {
            node_id: node for node_id, node in zip(bipartite_edge_ids, nodes)
            if node is not None
        }

    all_edges_data = []
    for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree):
        if v is None:
            continue
        # A1 Fix: If target is bipartite edge node, get content from node data
        target_id = k[1]
        if target_id.startswith("rel-") and target_id in bipartite_node_data:
            description = bipartite_node_data[target_id].get("content", target_id)
        else:
            # Entity node or fallback - use node ID
            description = target_id

        all_edges_data.append({
            "src_tgt": k,
            "rank": d,
            "description": description,
            **v
        })

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    vdb_bipartite_edges: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # Fixed: Actually query the vector database instead of assigning the object
    results = await vdb_bipartite_edges.query(keywords, top_k=query_param.top_k)

    if not results or not len(results):  # Check for None or empty
        return []  # Return empty list when no results (not empty strings)
    # Bug #4 Fix: Use defensive dict access to prevent KeyError
    # Extract edge names from query results (Bug #5 fix: use bipartite_edge_name, not hash ID)
    results = [r.get("bipartite_edge_name") for r in results if "bipartite_edge_name" in r]

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    # edge_degree = await asyncio.gather(
    #     *[knowledge_graph_inst.node_degree(r["bipartite_edge_name"]) for r in results]
    # )
    edge_datas = [
        {"bipartite_edge": k, "rank": v["weight"], **v}
        for k, v in zip(results, edge_datas)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    # Extract knowledge and source IDs together for evaluation
    knowledge_list = []
    for s in edge_datas:
        # A1 Fix: Extract content from node (hash IDs store content in 'content' attribute)
        # Fallback to hash ID for backward compatibility (though content should always exist)
        bipartite_edge_content = s.get("content", s["bipartite_edge"]).replace("<bipartite_edge>", "")
        # Extract source_ids from the edge (chunks where this edge appears)
        source_ids = []
        if "source_id" in s and s["source_id"]:
            # source_id may contain multiple IDs separated by GRAPH_FIELD_SEP
            source_ids = s["source_id"].split(GRAPH_FIELD_SEP) if isinstance(s["source_id"], str) else [s["source_id"]]
        knowledge_list.append((bipartite_edge_content, source_ids))
    return knowledge_list


async def _get_chunk_data(
    query: str,
    vdb_chunks: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    entity_source_ids: set,
    edge_source_ids: set,
    query_param: QueryParam,
):
    """
    Path C: Chunk-level retrieval (Phase 3.1 - Three-Path Retrieval)

    Combines two sources for 10 total candidate chunks:
    1. Direct vector search on vdb_chunks (top-5 chunks)
    2. Indirect extraction from Path A + Path B source_ids (another 5 chunks)

    This provides both semantic relevance (direct) and structural relevance (indirect).

    Args:
        query: User query string
        vdb_chunks: Chunk vector database
        text_chunks_db: Chunk metadata storage
        entity_source_ids: Set of chunk IDs from entity retrieval (Path A)
        edge_source_ids: Set of chunk IDs from edge retrieval (Path B)
        query_param: Query parameters

    Returns:
        List of tuples: (chunk_content, [source_id])
    """
    if vdb_chunks is None:
        logger.warning("[Path C] vdb_chunks is None, skipping chunk retrieval")
        return []

    chunk_candidates = []

    # Part 1: Direct vector search (top-5)
    try:
        direct_results = await vdb_chunks.query(query, top_k=5)
        if direct_results:
            for result in direct_results:
                chunk_id = result.get("id")
                if chunk_id:
                    chunk_data = await text_chunks_db.get_by_id(chunk_id)
                    if chunk_data and "content" in chunk_data:
                        chunk_candidates.append({
                            "content": chunk_data["content"],
                            "source_id": chunk_id,
                            "source": "direct_vector",
                            "score": result.get("score", 0.0),
                        })
            logger.info(f"[Path C] Found {len(chunk_candidates)} chunks via direct vector search")
    except Exception as e:
        logger.warning(f"[Path C] Direct vector search failed: {e}")

    # Part 2: Indirect extraction from Path A + Path B source_ids (top-5)
    indirect_source_ids = list(entity_source_ids.union(edge_source_ids))
    if indirect_source_ids:
        # Take top 5 from combined source IDs
        for chunk_id in indirect_source_ids[:5]:
            # Skip if already in direct results
            if any(c["source_id"] == chunk_id for c in chunk_candidates):
                continue

            chunk_data = await text_chunks_db.get_by_id(chunk_id)
            if chunk_data and "content" in chunk_data:
                chunk_candidates.append({
                    "content": chunk_data["content"],
                    "source_id": chunk_id,
                    "source": "indirect_graph",
                    "score": 0.5,  # Default score for indirect
                })

        logger.info(f"[Path C] Added {len([c for c in chunk_candidates if c['source'] == 'indirect_graph'])} indirect chunks from graph traversal")

    # Return full chunk candidates with metadata for proper scoring
    # Modified to preserve source type and original scores for weighted RRF
    logger.info(f"[Path C] Total: {len(chunk_candidates)} chunk candidates")
    return chunk_candidates


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["bipartite_edge"]) for edge in edge_datas]
    )
    
    entity_names = []
    seen = set()

    for node_data in node_datas:
        for e in node_data:
            if e[1] not in seen:
                entity_names.append(e[1])
                seen.add(e[1])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                # Only store valid data
                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    return combined_entities, combined_relationships, ""