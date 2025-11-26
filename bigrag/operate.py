import asyncio
import json
import os
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
    sanitize_extracted_text,         # Added for orphan node reduction
    fix_delimiter_corruption,        # Added for orphan node reduction
    description_quality_score,       # Added for orphan node reduction
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
# RRF (Reciprocal Rank Fusion) Scoring
# ========================

def rrf_score_fusion(rankings: list[list], k: int = 1) -> dict:
    """
    Reciprocal Rank Fusion (RRF) scoring for merging multiple ranked lists.

    Combines rankings by assigning scores based on reciprocal rank position.
    Items appearing in multiple lists accumulate higher scores.

    Args:
        rankings: List of ranking lists (each inner list is ordered by relevance)
                 Example: [["A", "B"], ["B", "C"]]
        k: Constant for RRF formula (default=1, matches current inline implementation)

    Returns:
        Dict mapping items to RRF scores (higher score = more relevant)

    Formula: score(item) = Σ 1/(rank + k) across all rankings where item appears

    Examples:
        >>> rrf_score_fusion([["A", "B"], ["B", "C"]])
        {"A": 1.0, "B": 1.5, "C": 0.5}  # B appears in both lists

        >>> rrf_score_fusion([])
        {}  # Empty input returns empty dict
    """
    scores = {}

    for ranking in rankings:
        for i, item in enumerate(ranking):
            if item not in scores:
                scores[item] = 0.0
            # RRF formula: 1/(rank + k) where rank is 0-indexed position
            scores[item] += 1.0 / (i + k)

    return scores


# ========================
# Entity Type Validation (A2)
# ========================

TYPE_NORMALIZATION_MAP = {
    # Teams & Organizations
    "TEAM": "organization",
    "CLUB": "organization",
    "GROUP": "organization",
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
    chunk_index = 0
    for start in range(0, len(tokens), max_token_size - overlap_token_size):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        chunk_content_stripped = chunk_content.strip()

        # Skip empty chunks (can happen with trailing whitespace tokens)
        if not chunk_content_stripped:
            continue

        chunk = {
            "tokens": min(max_token_size, len(tokens) - start),
            "content": chunk_content_stripped,
            "chunk_order_index": chunk_index,
        }
        # Preserve metadata if provided
        if doc_title:
            chunk["doc_title"] = doc_title
        if doc_metadata:
            chunk["doc_metadata"] = doc_metadata
        results.append(chunk)
        chunk_index += 1
    return results


def _normalize_entity_name(entity_name: str) -> str:
    """
    Normalize entity name based on script type (language-aware).

    Convention:
    - Latin scripts (English, Spanish, French, etc.): UPPERCASE for consistency
    - Non-Latin scripts (Bangla, Arabic, Hindi, Chinese, Japanese, etc.): Natural form

    This prevents breaking non-Latin entity names which don't have uppercase variants.

    Args:
        entity_name: Raw entity name from LLM extraction

    Returns:
        Normalized entity name (uppercase for Latin, natural for non-Latin)

    Examples:
        >>> _normalize_entity_name("albert einstein")
        "ALBERT EINSTEIN"

        >>> _normalize_entity_name("আইনস্টাইন")  # Bangla
        "আইনস্টাইন"

        >>> _normalize_entity_name("爱因斯坦")  # Chinese
        "爱因斯坦"
    """
    # Define Unicode ranges for non-Latin scripts
    NON_LATIN_RANGES = [
        (0x0980, 0x09FF),  # Bangla/Bengali
        (0x0600, 0x06FF),  # Arabic
        (0x0900, 0x097F),  # Devanagari (Hindi, Sanskrit, etc.)
        (0x0C00, 0x0C7F),  # Telugu
        (0x0B80, 0x0BFF),  # Tamil
        (0x0A80, 0x0AFF),  # Gujarati
        (0x0D00, 0x0D7F),  # Malayalam
        (0x0B00, 0x0B7F),  # Oriya
        (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
        (0x0C80, 0x0CFF),  # Kannada
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs (Chinese)
        (0x3040, 0x309F),  # Hiragana (Japanese)
        (0x30A0, 0x30FF),  # Katakana (Japanese)
        (0xAC00, 0xD7AF),  # Hangul (Korean)
        (0x0E00, 0x0E7F),  # Thai
        (0x1780, 0x17FF),  # Khmer
        (0x0F00, 0x0FFF),  # Tibetan
        (0x1000, 0x109F),  # Myanmar/Burmese
    ]

    # Check if entity name contains non-Latin characters
    for char in entity_name:
        char_code = ord(char)
        for start, end in NON_LATIN_RANGES:
            if start <= char_code <= end:
                # Found non-Latin character, return as-is
                return entity_name

    # Only Latin characters found, apply UPPERCASE convention
    return entity_name.upper()


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
    """
    Extract and validate a single entity from LLM output.

    Validation:
    - Exactly 5 fields (not <5, not >5)
    - Must have relation context (prevent orphan entities)
    - Sanitize all fields
    - Validate entity name/type/description

    Returns:
        dict or None: Entity data if valid, None if invalid
    """

    # DEBUG: Log input
    logger.debug(
        f"{chunk_key}: ENTITY INPUT: {len(record_attributes)} fields, "
        f"first={repr(record_attributes[0][:20] if record_attributes else 'EMPTY')}"
    )

    # Validate field count (EXACT, not >=)
    if len(record_attributes) != 5:
        if len(record_attributes) > 1 and '"entity"' in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: [REJECT] Entity has {len(record_attributes)}/5 fields "
                f"(expected exactly 5). Entity: {record_attributes[1] if len(record_attributes) > 1 else 'N/A'}"
            )
        else:
            logger.debug(f"{chunk_key}: [REJECT] Entity field count {len(record_attributes)} != 5")
        return None

    # Validate first field is "entity"
    if record_attributes[0] != '"entity"':
        logger.debug(
            f"{chunk_key}: [REJECT] First field is {repr(record_attributes[0][:20])}, expected '\"entity\"'"
        )
        return None

    # Validate relation context exists (prevent orphan entities)
    if not now_hyper_relation or now_hyper_relation == "":
        logger.warning(
            f"{chunk_key}: Entity extracted without relation context. "
            f"Creating default relation to prevent data loss. Entity: {record_attributes[1]}"
        )
        # Create a default relation for this chunk to link orphan entities
        # This prevents data loss while still tracking the sequencing issue
        from .constants import RELATION_PREFIX
        default_relation_content = f"General context for chunk {chunk_key}"
        # FIX: Use .strip() for consistency (though f-strings don't add whitespace)
        now_hyper_relation = compute_mdhash_id(default_relation_content.strip(), prefix=RELATION_PREFIX)
        # Note: The default relation won't be stored in maybe_edges,
        # but entities will have a valid hyper_relation reference

    # Sanitize entity name
    entity_name_raw = record_attributes[1]
    entity_name = sanitize_extracted_text(entity_name_raw, "entity_name")
    logger.debug(f"{chunk_key}: Entity name: '{entity_name_raw[:30]}' → '{entity_name[:30] if entity_name else 'EMPTY'}'")

    if not entity_name:
        logger.warning(
            f"{chunk_key}: [REJECT] Entity name became empty after sanitization. "
            f"Raw: '{entity_name_raw}'"
        )
        return None

    # Apply BiG-RAG convention: UPPERCASE entity names (language-aware)
    # Only uppercase Latin scripts; keep non-Latin scripts (Bangla, Arabic, Chinese, etc.) in natural form
    entity_name = _normalize_entity_name(entity_name)

    # Sanitize entity type
    entity_type_raw = record_attributes[2]
    entity_type = sanitize_extracted_text(entity_type_raw, "entity_type")
    logger.debug(f"{chunk_key}: Entity type: '{entity_type_raw}' → '{entity_type}'")

    if not entity_type:
        logger.warning(
            f"{chunk_key}: [REJECT] Entity type invalid for entity '{entity_name}'. "
            f"Raw: '{entity_type_raw}'"
        )
        return None

    # Normalize entity type (e.g., "PERSON" → "person", "Organization" → "organization")
    entity_type = normalize_entity_type(entity_type)

    # Sanitize description
    description_raw = record_attributes[3]
    description = sanitize_extracted_text(description_raw, "description")
    logger.debug(f"{chunk_key}: Description: {len(description_raw)} → {len(description)} chars")

    if not description:
        logger.warning(
            f"{chunk_key}: [REJECT] Description empty for entity '{entity_name}' of type '{entity_type}'"
        )
        return None

    # Parse weight (key score)
    try:
        weight = float(record_attributes[4])
        # Validate reasonable range
        if weight < 0 or weight > 100:
            logger.warning(
                f"{chunk_key}: Entity weight out of range (0-100): {weight} for '{entity_name}'. "
                f"Using clamped value."
            )
            weight = max(0, min(100, weight))
    except (ValueError, IndexError) as e:
        logger.warning(
            f"{chunk_key}: Invalid weight for entity '{entity_name}': {record_attributes[4]}. "
            f"Using default 50.0. Error: {e}"
        )
        weight = 50.0  # Default fallback

    # Return validated entity data
    logger.debug(f"{chunk_key}: [SUCCESS] Entity '{entity_name}' ({entity_type}) weight={weight}")
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=chunk_key,
        weight=weight,
        hyper_relation=now_hyper_relation,  # Link to parent relation
    )


async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
    Extract and validate a single relation from LLM output.

    Validation:
    - Exactly 3 fields (not <3, not >3)
    - Sanitize content
    - Validate completeness score range

    Returns:
        dict or None: Relation data if valid, None if invalid
    """

    # DEBUG: Log input
    logger.debug(
        f"{chunk_key}: RELATION INPUT: {len(record_attributes)} fields, "
        f"first={repr(record_attributes[0][:20] if record_attributes else 'EMPTY')}"
    )

    # Validate field count (EXACT, not >=)
    if len(record_attributes) != 3:
        if len(record_attributes) > 1 and '"relation"' in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: [REJECT] Relation has {len(record_attributes)}/3 fields (expected exactly 3)"
            )
        else:
            logger.debug(f"{chunk_key}: [REJECT] Relation field count {len(record_attributes)} != 3")
        return None

    # Validate first field is "relation"
    if record_attributes[0] != '"relation"':
        logger.debug(
            f"{chunk_key}: [REJECT] First field is {repr(record_attributes[0][:20])}, expected '\"relation\"'"
        )
        return None

    # Sanitize knowledge fragment (relation content)
    knowledge_fragment_raw = record_attributes[1]
    knowledge_fragment = sanitize_extracted_text(knowledge_fragment_raw, "relation")
    logger.debug(f"{chunk_key}: Relation content: {len(knowledge_fragment_raw)} → {len(knowledge_fragment)} chars")

    if not knowledge_fragment:
        logger.warning(
            f"{chunk_key}: [REJECT] Relation content became empty after sanitization. "
            f"Raw: '{knowledge_fragment_raw[:50]}...'"
        )
        return None

    # Parse completeness score
    try:
        weight = float(record_attributes[2])
        logger.debug(f"{chunk_key}: Relation score: {record_attributes[2]} → {weight}")
        # Validate reasonable range
        if weight < 0 or weight > 10:
            logger.warning(
                f"{chunk_key}: Relation completeness score out of range (0-10): {weight}. "
                f"Using clamped value."
            )
            weight = max(0, min(10, weight))
    except (ValueError, IndexError) as e:
        logger.warning(
            f"{chunk_key}: Invalid completeness score: {record_attributes[2]}. "
            f"Using default 5.0. Error: {e}"
        )
        weight = 5.0  # Default fallback

    # Generate hash-based ID for relation node
    from .constants import RELATION_PREFIX
    # FIX: Use .strip() for consistent relation ID generation
    edge_id = compute_mdhash_id(knowledge_fragment.strip(), prefix=RELATION_PREFIX)

    # Return validated relation data
    logger.debug(f"{chunk_key}: [SUCCESS] Relation '{knowledge_fragment[:40]}...' score={weight}")
    return dict(
        hyper_relation=edge_id,                # Relation node ID (hash-based)
        hyper_relation_content=knowledge_fragment,  # Actual content
        weight=weight,                         # Completeness score
        source_id=chunk_key,                   # Which chunk this came from
    )
    

async def _merge_relations_then_upsert(
    relation_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Merge and upsert relation nodes with weight aggregation.

    A1: Now accepts hash-based IDs and stores content as node attribute.

    Weight Semantics (A3):
    - weight = sum of completeness scores across all occurrences
    - Range: 0 to N×10 (where N = number of chunks mentioning this relation)
    - Higher weight = more frequently mentioned + higher completeness
    - No normalization (intentional - preserves frequency signal)

    Args:
        relation_name: Hash ID of the edge (e.g., "rel-abc123...")
        nodes_data: List of dicts with hyper_relation_content, weight, source_id
        knowledge_graph_inst: Graph storage instance
        global_config: Configuration dict

    Returns:
        Node data dict with relation_name
    """
    already_weights = []
    already_source_ids = []

    already_relation = await knowledge_graph_inst.get_node(relation_name)
    if already_relation is not None:
        already_weights.append(already_relation["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_relation["source_id"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    # A1: Extract content from first occurrence (all should be same since same hash)
    # Content is now stored as node attribute instead of being the ID
    content = nodes_data[0].get("hyper_relation_content", "") if nodes_data else ""

    node_data = dict(
        role="relation",
        content=content,  # A1: Store content as attribute
        weight=weight,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        relation_name,
        node_data=node_data,
    )
    node_data["relation_name"] = relation_name
    node_data["relation_content"] = content  # For VDB upsertion
    return node_data


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Merge and upsert entity nodes with weight aggregation.

    UNIFIED ENTITY ID STRATEGY:
    - Uses compute_mdhash_id() to generate stable entity_id from entity_name
    - entity_id format: "entity-abc123" (hash-based, stable across pipelines)
    - Replaces old name-based node IDs for consistency with production pipeline
    - Both entity_id and entity_name stored in node data

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
        Node data dict with entity_name and entity_id
    """
    # UNIFIED: Generate stable entity_id (same strategy as production pipeline)
    entity_id = compute_mdhash_id(entity_name, prefix="entity-")

    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_weights = []

    # UNIFIED: Use entity_id as node ID (not entity_name)
    already_node = await knowledge_graph_inst.get_node(entity_id)
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
    # Aggregate weights from all occurrences (same as relations)
    weight = sum([dp.get("weight", 0) for dp in nodes_data] + already_weights)

    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        role="entity",
        entity_name=entity_name,  # UNIFIED: Store entity_name in node data
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        weight=weight,
    )
    # UNIFIED: Use entity_id as node ID (not entity_name)
    await knowledge_graph_inst.upsert_node(
        entity_id,
        node_data=node_data,
    )
    node_data["entity_id"] = entity_id  # UNIFIED: Return entity_id for indexing
    return node_data


async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    # UNIFIED: Generate stable entity_id (same as _merge_nodes_then_upsert)
    entity_id = compute_mdhash_id(entity_name, prefix="entity-")

    edge_data = []

    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]

        already_weights = []
        already_source_ids = []

        # UNIFIED: Use entity_id for edge connections (not entity_name)
        if await knowledge_graph_inst.has_edge(hyper_relation, entity_id):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_id)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
        
        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(
            set([source_id] + already_source_ids)
        )

        # UNIFIED: Use entity_id for edge target (not entity_name)
        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_id,
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
    vdb_relations: BaseVectorStorage,
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

    def validate_extraction_results(
        maybe_nodes: dict,
        maybe_edges: dict,
        chunk_key: str
    ) -> dict:
        """
        Validate extraction results and detect potential orphan relations.

        Purpose: Catch orphan relations early (at extraction time) before they're
                 stored in the graph, allowing for corrective action or warnings.

        Args:
            maybe_nodes: dict[entity_name, list[entity_data]]
            maybe_edges: dict[relation_id, list[relation_data]]
            chunk_key: Chunk identifier for logging

        Returns:
            dict: {
                "total_entities": int,
                "total_relations": int,
                "orphan_relations": list[str],  # Relation IDs without entities
                "warnings": list[str]
            }
        """
        validation_report = {
            "total_entities": len(maybe_nodes),
            "total_relations": len(maybe_edges),
            "orphan_relations": [],
            "warnings": []
        }

        # Check each relation for linked entities
        for relation_id, relation_list in maybe_edges.items():
            relation_content = relation_list[0].get("hyper_relation_content", "")

            # Find entities that reference this relation
            linked_entities = []
            for entity_name, entity_list in maybe_nodes.items():
                entity_relation = entity_list[0].get("hyper_relation", "")
                if entity_relation == relation_id:
                    linked_entities.append(entity_name)

            if len(linked_entities) == 0:
                # Orphan relation detected!
                validation_report["orphan_relations"].append(relation_id)

                # Log warning with truncated content
                display_content = relation_content[:80] + "..." if len(relation_content) > 80 else relation_content
                warning = (
                    f"ORPHAN RELATION (no entities): '{display_content}'"
                )
                validation_report["warnings"].append(warning)
                logger.warning(f"{chunk_key}: {warning}")

        # Log summary
        if validation_report["orphan_relations"]:
            orphan_count = len(validation_report["orphan_relations"])
            total_count = validation_report["total_relations"]
            orphan_rate = orphan_count / total_count if total_count > 0 else 0

            logger.warning(
                f"{chunk_key}: Found {orphan_count} orphan relations "
                f"out of {total_count} total relations ({orphan_rate:.1%})"
            )

        return validation_report

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        # Extract metadata for context enhancement (Phase 2.1 improvement)
        doc_title = chunk_dp.get("doc_title", "")
        doc_metadata = chunk_dp.get("doc_metadata", {})

        # Build context-enriched input text with bracket-style formatting
        # (Phase 3.1: Adopted from LightRAG to prevent metadata confusion)
        context_parts = []

        if doc_title:
            context_parts.append(f"Title: {doc_title}")

        if doc_metadata:
            metadata_str = ", ".join(
                f"{k}: {v}" for k, v in doc_metadata.items()
                if k != "title" and v  # Skip empty values and title (already included)
            )
            if metadata_str:
                context_parts.append(f"Metadata: {metadata_str}")

        # Combine with bracket markers for clear semantic boundaries
        if context_parts:
            metadata_block = "\n".join(context_parts)
            enriched_content = (
                f"[DOCUMENT CONTEXT]\n"
                f"{metadata_block}\n\n"
                f"[CHUNK CONTENT]\n"
                f"{content}"
            )
        else:
            # No metadata available, use content as-is
            enriched_content = content

        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=enriched_content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=enriched_content)

        # B5: Use semaphore to limit concurrent LLM calls
        async with llm_semaphore:
            final_result = await use_llm_func(hint_prompt)

        # DEBUG: Log raw LLM response
        logger.debug(f"{chunk_key}: LLM response length: {len(final_result)} chars")
        logger.debug(f"{chunk_key}: LLM response preview: {final_result[:200] if final_result else 'EMPTY'}...")
        print(f"\n[DEBUG] {chunk_key}: LLM response length: {len(final_result)} chars")

        # Fix corrupted delimiters BEFORE parsing
        # (LLM sometimes outputs <> instead of <|>, || instead of <|>, etc.)
        final_result = fix_delimiter_corruption(
            final_result,
            context_base["tuple_delimiter"]
        )

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        # Parse initial extraction results FIRST
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        # DEBUG: Log parsed records count
        logger.debug(f"{chunk_key}: Parsed {len(records)} records from LLM response")

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation=""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)

            # Fix delimiter corruption on individual record (CRITICAL FIX)
            record = fix_delimiter_corruption(record, context_base["tuple_delimiter"])

            # DEBUG: Print delimiter being used
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

        # Gleaning loop with smart merge (quality-based)
        for now_glean_index in range(entity_extract_max_gleaning):
            async with llm_semaphore:
                glean_result = await use_llm_func(continue_prompt, history_messages=history)

            # Fix corrupted delimiters in gleaning result
            glean_result = fix_delimiter_corruption(
                glean_result,
                context_base["tuple_delimiter"]
            )

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)

            # Parse gleaning results into separate structures
            glean_records = split_string_by_multi_markers(
                glean_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )

            maybe_glean_nodes = defaultdict(list)
            maybe_glean_edges = defaultdict(list)
            glean_hyper_relation = now_hyper_relation  # Carry forward last relation context

            # Parse each gleaning record
            for record in glean_records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(
                    record, [context_base["tuple_delimiter"]]
                )

                # Try parsing as relation
                if_relation = await _handle_single_hyperrelation_extraction(
                    record_attributes, chunk_key
                )
                if if_relation is not None:
                    relation_id = if_relation["hyper_relation"]
                    maybe_glean_edges[relation_id].append(if_relation)
                    glean_hyper_relation = relation_id

                # Try parsing as entity
                if_entities = await _handle_single_entity_extraction(
                    record_attributes, chunk_key, glean_hyper_relation
                )
                if if_entities is not None:
                    entity_name = if_entities["entity_name"]
                    maybe_glean_nodes[entity_name].append(if_entities)

            # SMART MERGE: Compare quality and keep better version
            # Merge entities
            for entity_name, glean_entity_list in maybe_glean_nodes.items():
                if entity_name in maybe_nodes:
                    # Entity already exists - compare descriptions
                    original_desc = maybe_nodes[entity_name][0].get("description", "")
                    glean_desc = glean_entity_list[0].get("description", "")

                    # Use quality scoring (considers length, keywords, completeness)
                    original_quality = description_quality_score(original_desc)
                    glean_quality = description_quality_score(glean_desc)

                    if glean_quality > original_quality:
                        logger.debug(
                            f"{chunk_key}: Gleaning improved entity '{entity_name}': "
                            f"quality {original_quality:.0f} → {glean_quality:.0f}"
                        )
                        maybe_nodes[entity_name] = glean_entity_list
                    else:
                        # Keep original (better quality)
                        logger.debug(
                            f"{chunk_key}: Keeping original entity '{entity_name}' "
                            f"(better quality: {original_quality:.0f} vs {glean_quality:.0f})"
                        )
                else:
                    # New entity from gleaning
                    logger.debug(f"{chunk_key}: Gleaning found new entity: '{entity_name}'")
                    maybe_nodes[entity_name] = glean_entity_list

            # Merge relations
            for relation_id, glean_relation_list in maybe_glean_edges.items():
                if relation_id in maybe_edges:
                    # Relation already exists - compare content quality
                    original_content = maybe_edges[relation_id][0].get("hyper_relation_content", "")
                    glean_content = glean_relation_list[0].get("hyper_relation_content", "")

                    original_quality = description_quality_score(original_content)
                    glean_quality = description_quality_score(glean_content)

                    if glean_quality > original_quality:
                        logger.debug(
                            f"{chunk_key}: Gleaning improved relation {relation_id[:16]}..."
                        )
                        maybe_edges[relation_id] = glean_relation_list
                else:
                    # New relation from gleaning
                    logger.debug(f"{chunk_key}: Gleaning found new relation: {relation_id[:16]}...")
                    maybe_edges[relation_id] = glean_relation_list

            # Check if should continue gleaning
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            async with llm_semaphore:
                if_loop_result: str = await use_llm_func(
                    if_loop_prompt, history_messages=history
                )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        # Validate extraction results (detect orphans early)
        validation_report = validate_extraction_results(
            maybe_nodes,
            maybe_edges,
            chunk_key
        )

        # If orphan rate is very high, log error
        if validation_report["total_relations"] > 0:
            orphan_rate = (
                len(validation_report["orphan_relations"]) /
                validation_report["total_relations"]
            )

            # Threshold: 10% (not 30% as in original plan)
            if orphan_rate > 0.10:
                logger.error(
                    f"{chunk_key}: HIGH ORPHAN RATE ({orphan_rate:.1%})! "
                    f"Expected <5%, found {len(validation_report['orphan_relations'])}/"
                    f"{validation_report['total_relations']} orphans. "
                    f"LLM may not be following extraction rules."
                )

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
            
    logger.info("Inserting relations into storage...")
    all_relations_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_relations_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting relations",
        unit="entity",
    ):
        all_relations_data.append(await result)
            
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

    if not len(all_relations_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relations and entities, maybe your LLM is not working"
        )
        return None

    if not len(all_relations_data):
        logger.warning("Didn't extract any relations")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    # B3: Wrap VDB operations with retry mechanism
    from .utils import safe_operation_with_retry
    from .constants import DEFAULT_MAX_RETRIES

    if vdb_relations is not None:
        # FIX #2: Clearer field naming
        # relation_name field contains hash ID (rel-abc123), so rename to relation_id
        data_for_vdb = {
            dp["relation_name"]: {  # Key: hash ID like "rel-abc123"
                "content": dp.get("relation_content", ""),  # Use actual content for embedding
                "relation_id": dp["relation_name"],  # FIX #2: Store hash ID with clear field name
            }
            for dp in all_relations_data
        }
        await safe_operation_with_retry(
            lambda: vdb_relations.upsert(data_for_vdb),
            "VDB upsert relations",
            context=f"{len(data_for_vdb)} edges",
            max_retries=global_config.get("api_retry_attempts", DEFAULT_MAX_RETRIES),
        )

    if vdb_entities is not None:
        # UNIFIED: Use entity_id as VDB key (same format as production pipeline: "entity-abc123")
        data_for_vdb = {
            dp["entity_id"]: {  # UNIFIED: Use entity_id returned from _merge_nodes_then_upsert
                "content": dp["entity_name"] + dp["description"],
                "entity_id": dp["entity_id"],  # UNIFIED: Store entity_id for retrieval
                "entity_name": dp["entity_name"],  # Keep for backward compatibility
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


async def preprocess_query(
    query: str,
    language: str,
    llm_func: callable,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[str, str]:
    """
    Preprocess query to generate normalized and statement forms.

    Returns:
        (normalized_query, statement_query)
    """
    # OPTIONAL: Check cache (can be implemented later)
    # if hashing_kv is not None:
    #     args_hash = compute_mdhash_id(query + language, prefix="query_preprocess-")
    #     cached_result = await hashing_kv.get_by_id(args_hash)
    #     if cached_result is not None:
    #         cached_data = json.loads(cached_result.get("return_response", "{}"))
    #         return (
    #             cached_data.get("normalized_query", query),
    #             cached_data.get("statement_query", query)
    #         )

    # Build prompt
    prompt = PROMPTS["query_preprocessing"].format(query=query, language=language)

    # Call LLM
    try:
        response = await llm_func(
            prompt,
            max_tokens=512,
            temperature=0.0,
        )

        # Parse JSON (handle potential markdown wrapping)
        response_text = response.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        result = json.loads(response_text)
        normalized_query = result.get("normalized_query", query)
        statement_query = result.get("statement_query", query)

        # OPTIONAL: Save to cache (can be implemented later)
        # if hashing_kv is not None:
        #     await hashing_kv.upsert({
        #         args_hash: {
        #             "query": query,
        #             "language": language,
        #             "return_response": json.dumps(result),
        #         }
        #     })

        # Print detailed comparison for debugging
        logger.info("=" * 80)
        logger.info("[Query Preprocessing] Comparison:")
        logger.info("-" * 80)
        logger.info(f"LANGUAGE: {language}")
        logger.info("-" * 80)
        logger.info(f"ORIGINAL QUERY:\n  {query}")
        logger.info("-" * 80)
        logger.info(f"NORMALIZED QUERY:\n  {normalized_query}")
        logger.info("-" * 80)
        logger.info(f"STATEMENT QUERY:\n  {statement_query}")
        logger.info("=" * 80)

        return normalized_query, statement_query

    except Exception as e:
        logger.warning(f"[Query Preprocess] Failed: {e}. Using raw query.")
        return query, query  # Graceful degradation


def _format_knowledge_as_structured(knowledge_list: list[dict]) -> str:
    """
    Convert structured knowledge list to formatted string with sections.

    Args:
        knowledge_list: List of dicts with structure:
            {
                "<knowledge>": "text content",
                "<coherence>": 0.95,
                "<source_ids>": ["id1", "id2"],
                "<type>": "entity" | "relation" | "chunk" | "chunk_reranked",
                "<metadata>": {"category": "...", "title": "...", "tags": [...]}  # optional
            }

    Returns:
        Formatted string with three sections: Entities, Relations, Chunks
    """
    if not knowledge_list:
        return "No relevant knowledge found."

    # Step 1: Separate by type
    entities = [k for k in knowledge_list if k.get("<type>") == "entity"]
    relations = [k for k in knowledge_list if k.get("<type>") == "relation"]
    chunks = [k for k in knowledge_list if k.get("<type>") in ["chunk_reranked", "chunk", "direct_vector", "indirect_graph"]]

    sections = []

    # Section 1: Entities
    if entities:
        entity_section = "### Knowledge Graph - Entities\n\n"
        for i, ent in enumerate(entities, 1):
            entity_section += f"{i}. {ent['<knowledge>']}\n"
            if ent.get("<coherence>") is not None:
                entity_section += f"   Relevance Score: {ent['<coherence>']:.2f}\n"
            if ent.get("<source_ids>"):
                sources = ent["<source_ids>"][:3]  # Limit to 3
                entity_section += f"   Sources: {', '.join(sources)}\n"
            entity_section += "\n"
        sections.append(entity_section)

    # Section 2: Relations
    if relations:
        relation_section = "### Knowledge Graph - Relations\n\n"
        for i, rel in enumerate(relations, 1):
            relation_section += f"{i}. {rel['<knowledge>']}\n"
            if rel.get("<coherence>") is not None:
                relation_section += f"   Relevance Score: {rel['<coherence>']:.2f}\n"
            if rel.get("<source_ids>"):
                sources = rel["<source_ids>"][:3]
                relation_section += f"   Sources: {', '.join(sources)}\n"
            relation_section += "\n"
        sections.append(relation_section)

    # Section 3: Document Chunks
    if chunks:
        chunk_section = "### Document Chunks\n\n"
        for i, chunk in enumerate(chunks, 1):
            chunk_section += f"{i}. {chunk['<knowledge>']}\n"

            # Add metadata if present
            if chunk.get("<metadata>"):
                meta = chunk["<metadata>"]
                meta_parts = []
                if meta.get("category"):
                    meta_parts.append(f"Category={meta['category']}")
                if meta.get("title"):
                    meta_parts.append(f"Title={meta['title']}")
                if meta.get("tags") and isinstance(meta["tags"], list):
                    meta_parts.append(f"Tags={','.join(meta['tags'][:3])}")
                if meta_parts:
                    chunk_section += f"   [Metadata: {', '.join(meta_parts)}]\n"

            # Add source reference
            if chunk.get("<source_ids>"):
                chunk_section += f"   Source: {chunk['<source_ids>'][0]}\n"
            chunk_section += "\n"
        sections.append(chunk_section)

    return "\n".join(sections).strip()


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    vdb_entities: BaseVectorStorage,  # Bug #6 Fix: Correct type annotation
    vdb_relations: BaseVectorStorage,  # Bug #6 Fix: Correct type annotation
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    vdb_chunks: BaseVectorStorage,  # Phase 3.2: Added vdb_chunks parameter
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> Union[str, list]:

    # Feature flag for easy rollback
    # Determine whether to enable preprocessing
    # Priority 1: Per-query override (query_param.enable_query_preprocessing)
    # Priority 2: Global config (ENABLE_QUERY_PREPROCESSING environment variable)
    if query_param.enable_query_preprocessing is not None:
        # Per-query override takes precedence
        enable_preprocessing = query_param.enable_query_preprocessing
        logger.info(f"[Query Preprocess] Using per-query override: enable_query_preprocessing={enable_preprocessing}")
    else:
        # Use global default from environment variable
        enable_preprocessing = os.getenv("ENABLE_QUERY_PREPROCESSING", "true").lower() == "true"

    if enable_preprocessing:
        # Preprocess query
        # Cascading language priority:
        # Priority 1: query_param.language (per-query override)
        # Priority 2: global_config["addon_params"]["language"] (from .env)
        # Priority 3: PROMPTS["DEFAULT_LANGUAGE"] (hardcoded fallback = "English")
        if query_param.language is not None:
            language = query_param.language
            logger.info(f"[Query Preprocess] Using per-query language override: {language}")
        else:
            language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])

        llm_func = global_config["llm_model_func"]
        normalized_query, statement_query = await preprocess_query(
            query=query,
            language=language,
            llm_func=llm_func,
            global_config=global_config,
            hashing_kv=hashing_kv,
        )
    else:
        # Preprocessing disabled: use raw query for both paths
        logger.info("[Query Preprocess] Preprocessing disabled, using raw query")
        normalized_query = query
        statement_query = query

    # Path A: Use normalized query (entity names + PRF)
    ll_keywords = normalized_query

    # Path B & C: Use statement query (knowledge segments + chunks)
    hl_keywords = statement_query

    keywords = [ll_keywords, hl_keywords]
    knowledge_list = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        vdb_entities,
        vdb_relations,
        text_chunks_db,
        vdb_chunks,  # Phase 3.2: Pass vdb_chunks
        query_param,
    )

    # Return structured list for API endpoints or formatted string for LLM context
    if query_param.only_need_context:
        return knowledge_list  # Return list of dicts for API endpoints
    else:
        return _format_knowledge_as_structured(knowledge_list)  # Return formatted string for LLM



async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    vdb_entities: BaseVectorStorage,
    vdb_relations: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    vdb_chunks: BaseVectorStorage,  # Phase 3.2: Added vdb_chunks parameter
    query_param: QueryParam,
):
    """
    Three-Path Retrieval Context Builder (Phase 3.2 Enhancement)

    Combines three retrieval paths:
    - Path A: Entity-based (structural, high-level)
    - Path B: Relation-based (relational, knowledge fragments)
    - Path C: Chunk-based (semantic, raw text)

    Returns 5 structured knowledge items + 5 chunk items = 10 total context items
    """

    ll_keywords, hl_keywords = query[0], query[1]

    # Path A: Entity retrieval
    knowledge_list_1 = await _get_node_data(
        ll_keywords,
        knowledge_graph_inst,
        vdb_entities,
        text_chunks_db,
        query_param,
    )

    # Path B: Relation retrieval
    knowledge_list_2 = await _get_edge_data(
        hl_keywords,
        knowledge_graph_inst,
        vdb_relations,
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
        hl_keywords,  # Use statement query for chunk search
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
            # Phase 1 Fix: Distinguish entities from relations by prefix
            if k.startswith("ENTITY:"):
                know_type[k] = "entity"
            else:
                know_type[k] = "relation"  # Relations from entity traversal
        score = 1/(i+1)
        know_score[k] += score
        if source_ids:
            know_sources[k].update(source_ids if isinstance(source_ids, (list, set)) else [source_ids])

    # Path B contributions
    for i, (k, source_ids) in enumerate(knowledge_list_2):
        if k not in know_score:
            know_score[k] = 0
            know_sources[k] = set()
            know_type[k] = "relation"
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

        chunk_dict = {
            "content": chunk["content"],
            "score": score,
            "sources": [chunk["source_id"]],
            "type": chunk["source"]  # Preserve source type for debugging
        }

        # BUG FIX: Preserve metadata from _get_chunk_data()
        if chunk.get("metadata"):
            chunk_dict["metadata"] = chunk["metadata"]

        chunk_knowledge.append(chunk_dict)

    # Phase 3.4: Apply semantic reranking to chunks if enabled
    if query_param.enable_reranking and chunk_knowledge:
        try:
            from .reranker import rerank_chunks
            # Prepare chunks for reranking: (content, source_ids, metadata)
            chunk_candidates = [
                (c["content"], c["sources"], c.get("metadata"))
                for c in chunk_knowledge
            ]
            # Rerank and get top-5
            reranked = await rerank_chunks(
                query=ll_keywords,  # Use normalized query (same as entity search)
                chunks=chunk_candidates,
                top_k=5,
                use_reranking=True
            )
            # Update chunk_knowledge with reranked results and scores
            # reranked is a list of dicts: [{"content": str, "sources": list, "score": float, "metadata": dict (optional)}, ...]
            chunk_knowledge = [
                {
                    "content": item["content"],
                    "score": item["score"],
                    "sources": item["sources"],
                    "type": "chunk_reranked",
                    **({'metadata': item['metadata']} if item.get('metadata') else {})
                }
                for item in reranked
            ]
            logger.info("[Reranking] Applied cross-encoder reranking to chunks")
        except Exception as e:
            logger.warning(f"[Reranking] Failed, using original ranking: {e}")
            # Keep original chunk_knowledge

    # Take top-N structured knowledge (from Path A + B) - configurable via num_kg_in_context
    structured_knowledge = sorted(know_score.items(), key=lambda x: x[1], reverse=True)[:query_param.num_kg_in_context]

    # Combine: N KG items + M chunks (default: 15 + 5 = 20 total)
    knowledge = []

    # Add structured knowledge (relations only - entities disabled)
    for k, score in structured_knowledge:
        sources = list(know_sources.get(k, []))
        knowledge.append({
            "<knowledge>": k,
            "<coherence>": round(score, 3),
            "<source_ids>": sources,
            "<type>": know_type.get(k, "unknown")  # Add type for debugging
        })

    # Add chunk knowledge - configurable via num_chunks_in_context
    for chunk in chunk_knowledge[:query_param.num_chunks_in_context]:
        chunk_item = {
            "<knowledge>": chunk["content"],
            "<coherence>": round(chunk["score"], 3),
            "<source_ids>": chunk["sources"],
            "<type>": chunk["type"]
        }

        # NEW: Add metadata if present
        if chunk.get("metadata"):
            chunk_item["<metadata>"] = chunk["metadata"]

        knowledge.append(chunk_item)

    logger.info(f"[Three-Path Retrieval] Returning {len(knowledge)} items: "
                f"{len(structured_knowledge)} relations (Path A + Path B combined via RRF) + "
                f"{min(len(chunk_knowledge), query_param.num_chunks_in_context)} chunks (Path C)")

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
    # CRITICAL FIX (Jan 2025): Extract entity IDs from VDB results
    # VDB now returns: {"__id__": "entity-abc123", "id": "entity-abc123", "entity_id": "entity-abc123", "entity_name": "name", ...}
    # Priority: entity_id (new field from Fix #1) > __id__ > id (all contain the same hash ID)
    results = [r.get("entity_id", r.get("__id__", r.get("id"))) for r in results]
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
        {**n, "entity_id": k, "rank": d}  # Option B3: Store entity_id instead of overwriting entity_name
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    # Phase 1: DISABLED - Using only relation descriptions (like GraphR1)
    # Following GraphR1's approach: focus on knowledge fragments (relations) instead of entity descriptions
    # entity_knowledge_list = []
    # for i, entity in enumerate(node_datas[:10]):  # Top-10 most relevant entities
    #     if entity and "description" in entity:
    #         # Format: "ENTITY: {name} ({type}) - {description}"
    #         entity_desc = (
    #             f"ENTITY: {entity['entity_name']} "
    #             f"({entity.get('entity_type', 'unknown')}) - "
    #             f"{entity['description']}"
    #         )
    #         # Extract source_ids from entity (chunks where this entity appears)
    #         source_ids = []
    #         if "source_id" in entity and entity["source_id"]:
    #             source_ids = (
    #                 entity["source_id"].split(GRAPH_FIELD_SEP)
    #                 if isinstance(entity["source_id"], str)
    #                 else [entity["source_id"]]
    #             )
    #         entity_knowledge_list.append((entity_desc, source_ids))

    # Empty list - only use relations from Path A
    entity_knowledge_list = []

    # Get relations connected to all entities (unchanged)
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )

    # Extract relation knowledge and source IDs
    relation_knowledge_list = []
    for s in use_relations:
        # A1 Fix: Description now contains actual content from _find_most_related_edges_from_entities
        description = s["description"]
        # Extract source_ids from the relation (chunks where this relation appears)
        source_ids = []
        if "source_id" in s and s["source_id"]:
            # source_id may contain multiple IDs separated by GRAPH_FIELD_SEP
            source_ids = s["source_id"].split(GRAPH_FIELD_SEP) if isinstance(s["source_id"], str) else [s["source_id"]]
        relation_knowledge_list.append((description, source_ids))

    # Combine: 10 entity descriptions + all relation descriptions
    # This provides balanced entity-level and relation-level context for RRF
    knowledge_list = entity_knowledge_list + relation_knowledge_list

    logger.info(f"[Path A] Returning 0 entity descriptions (disabled) + {len(relation_knowledge_list)} relations")
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
    # FIX #3: Use entity_id instead of entity_name (graph nodes indexed by entity_id)
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_id"]) for dp in node_datas]
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
    """
    Find relations connected to entities via multi-hop graph traversal.

    Phase 2: Implements static multi-hop reasoning (1-3 hops) based on query_param.max_hops.
    Bipartite structure ensures alternating traversal: Entity → Relation → Entity → Relation

    Args:
        node_datas: Initial entity nodes from vector search
        query_param: Query parameters including max_hops
        knowledge_graph_inst: Graph storage instance

    Returns:
        List of relation dictionaries sorted by (rank, weight), each containing:
        - src_tgt: (source_id, target_id) tuple
        - rank: Edge degree (how many times this edge appears)
        - description: Relation content text
        - source_id: Chunks where this relation appears
        - weight: Aggregated importance score
        - hop: Which hop this relation was discovered in (for debugging)
    """
    # Initialize traversal state
    all_relations = []
    # CRITICAL FIX (Jan 2025): Use entity_id instead of entity_name
    # Graph nodes are indexed by entity_id (entity-abc123), not entity_name
    current_entities = {dp["entity_id"]: dp for dp in node_datas}
    visited_entities = set(current_entities.keys())

    logger.info(f"[Multi-Hop] Starting traversal with {len(current_entities)} seed entities, max_hops={query_param.max_hops}")

    # Multi-hop traversal loop
    for hop in range(query_param.max_hops):
        logger.info(f"[Multi-Hop] Hop {hop+1}/{query_param.max_hops}: Processing {len(current_entities)} entities")

        # Get all edges from current hop entities
        # Use entity_id (not entity_name) to query graph
        edges_batch = await asyncio.gather(
            *[knowledge_graph_inst.get_node_edges(entity_id)
              for entity_id in current_entities.keys()]
        )

        # Collect unique edges
        all_edges = []
        seen_edges = set()
        for edges in edges_batch:
            for e in edges:
                edge_tuple = tuple(e)
                if edge_tuple not in seen_edges:
                    seen_edges.add(edge_tuple)
                    all_edges.append(edge_tuple)

        if not all_edges:
            logger.info(f"[Multi-Hop] No edges found at hop {hop+1}, stopping early")
            break

        # Fetch edge metadata and degrees in parallel
        all_edges_pack = await asyncio.gather(
            *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
        )
        all_edges_degree = await asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
        )

        # Fetch relation node data for content extraction
        relation_ids = [e[1] for e in all_edges if e[1].startswith("rel-")]
        relation_node_data = {}
        if relation_ids:
            nodes = await asyncio.gather(
                *[knowledge_graph_inst.get_node(node_id) for node_id in relation_ids]
            )
            relation_node_data = {
                node_id: node for node_id, node in zip(relation_ids, nodes)
                if node is not None
            }

        # Process edges and prepare for next hop
        hop_relations = []
        next_entities = {}

        for edge, edge_data, edge_degree in zip(all_edges, all_edges_pack, all_edges_degree):
            if edge_data is None:
                continue

            src, tgt = edge

            # Extract relation content
            if tgt.startswith("rel-"):  # Target is a relation node
                relation_node = relation_node_data.get(tgt)
                if relation_node:
                    hop_relations.append({
                        "src_tgt": edge,
                        "rank": edge_degree,
                        "description": relation_node.get("content", tgt),
                        "source_id": relation_node.get("source_id", ""),
                        "weight": relation_node.get("weight", 0),
                        "hop": hop + 1  # Track which hop discovered this relation
                    })

                    # If more hops needed, collect entities connected through this relation
                    if hop < query_param.max_hops - 1:
                        # Get entities connected to this relation node
                        relation_edges = await knowledge_graph_inst.get_node_edges(tgt)
                        for rel_edge in relation_edges:
                            connected_entity = rel_edge[1]
                            # Only traverse to unvisited entity nodes
                            if not connected_entity.startswith("rel-") and connected_entity not in visited_entities:
                                entity_data = await knowledge_graph_inst.get_node(connected_entity)
                                if entity_data:
                                    next_entities[connected_entity] = entity_data
                                    visited_entities.add(connected_entity)

        all_relations.extend(hop_relations)
        logger.info(f"[Multi-Hop] Hop {hop+1} collected {len(hop_relations)} relations, discovered {len(next_entities)} new entities")

        # Prepare for next hop
        if hop < query_param.max_hops - 1:
            current_entities = next_entities
            if not current_entities:
                logger.info(f"[Multi-Hop] Early stop at hop {hop+1}: No more entities to traverse")
                break

    # Sort all collected relations by (rank, weight) - higher is better
    all_relations.sort(key=lambda x: (x["rank"], x["weight"]), reverse=True)
    logger.info(f"[Multi-Hop] Total collected: {len(all_relations)} relations from {hop+1} hops")

    return all_relations


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    vdb_relations: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # Fixed: Actually query the vector database instead of assigning the object
    results = await vdb_relations.query(keywords, top_k=query_param.top_k)

    if not results or not len(results):  # Check for None or empty
        return []  # Return empty list when no results (not empty strings)
    # CRITICAL FIX (Jan 2025): Extract relation IDs from VDB results
    # VDB now returns: {"__id__": "rel-abc123", "id": "rel-abc123", "relation_id": "rel-abc123", ...}
    # Priority: relation_id (new field) > __id__ > id (all contain the same hash ID)
    results = [r.get("relation_id", r.get("__id__", r.get("id"))) for r in results]

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    # edge_degree = await asyncio.gather(
    #     *[knowledge_graph_inst.node_degree(r["relation_name"]) for r in results]
    # )
    edge_datas = [
        {"relation": k, "rank": v["weight"], **v}
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
        relation_content = s.get("content", s["relation"]).replace("<relation>", "")
        # Extract source_ids from the edge (chunks where this edge appears)
        source_ids = []
        if "source_id" in s and s["source_id"]:
            # source_id may contain multiple IDs separated by GRAPH_FIELD_SEP
            source_ids = s["source_id"].split(GRAPH_FIELD_SEP) if isinstance(s["source_id"], str) else [s["source_id"]]
        knowledge_list.append((relation_content, source_ids))

    logger.info(f"[Path B] Found {len(knowledge_list)} relations via direct vector search")
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
                        chunk_dict = {
                            "content": chunk_data["content"],
                            "source_id": chunk_id,
                            "source": "direct_vector",
                            "score": result.get("score", 0.0),
                        }

                        # NEW: Extract metadata from chunk_data
                        metadata = {}
                        if chunk_data.get("doc_title"):
                            metadata["title"] = chunk_data["doc_title"]

                        if chunk_data.get("doc_metadata"):
                            doc_meta = chunk_data["doc_metadata"]
                            if isinstance(doc_meta, dict):
                                if doc_meta.get("category"):
                                    metadata["category"] = doc_meta["category"]
                                if doc_meta.get("tags"):
                                    metadata["tags"] = doc_meta["tags"]
                                # Add other fields as needed
                                for key in ["department", "author", "date"]:
                                    if doc_meta.get(key):
                                        metadata[key] = doc_meta[key]

                        if metadata:
                            chunk_dict["metadata"] = metadata

                        chunk_candidates.append(chunk_dict)
            logger.info(f"[Path C] Found {len(chunk_candidates)} chunks via direct vector search")
    except Exception as e:
        logger.warning(f"[Path C] Direct vector search failed: {e}")

    # Part 2: Indirect extraction from Path A + Path B source_ids (top-15)
    # Increased from 5 to 15 to match increased KG context (15 relations)
    indirect_source_ids = list(entity_source_ids.union(edge_source_ids))
    if indirect_source_ids:
        # Take top 15 from combined source IDs (increased candidate pool)
        for chunk_id in indirect_source_ids[:15]:
            # Skip if already in direct results
            if any(c["source_id"] == chunk_id for c in chunk_candidates):
                continue

            chunk_data = await text_chunks_db.get_by_id(chunk_id)
            if chunk_data and "content" in chunk_data:
                chunk_dict = {
                    "content": chunk_data["content"],
                    "source_id": chunk_id,
                    "source": "indirect_graph",
                    "score": 0.5,  # Default score for indirect
                }

                # NEW: Extract metadata from chunk_data (SAME pattern as direct chunks)
                metadata = {}
                if chunk_data.get("doc_title"):
                    metadata["title"] = chunk_data["doc_title"]

                if chunk_data.get("doc_metadata"):
                    doc_meta = chunk_data["doc_metadata"]
                    if isinstance(doc_meta, dict):
                        if doc_meta.get("category"):
                            metadata["category"] = doc_meta["category"]
                        if doc_meta.get("tags"):
                            metadata["tags"] = doc_meta["tags"]
                        # Add other fields as needed
                        for key in ["department", "author", "date"]:
                            if doc_meta.get(key):
                                metadata[key] = doc_meta[key]

                if metadata:
                    chunk_dict["metadata"] = metadata

                chunk_candidates.append(chunk_dict)

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
        *[knowledge_graph_inst.get_node_edges(edge["relation"]) for edge in edge_datas]
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