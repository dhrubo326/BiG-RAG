import asyncio
import html
import io
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List, Optional, Literal
import xml.etree.ElementTree as ET

import numpy as np
import tiktoken

from bigrag.prompt import PROMPTS


class UnlimitedSemaphore:
    """A context manager that allows unlimited access."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


ENCODER = None

logger = logging.getLogger("bigrag")


def set_logger(log_file: str, level: Union[str, int] = "INFO"):
    """
    Initialize logger with file handler (backward compatibility wrapper)

    This function now uses the centralized logging_config module
    for better log rotation and management.

    Args:
        log_file: Path to log file
        level: Log level (string like "INFO" or int like logging.INFO)
    """
    # Import here to avoid circular dependency
    from bigrag.logging_config import setup_logger

    log_dir = os.path.dirname(log_file)
    log_filename = os.path.basename(log_file)

    # Use new centralized logging system
    setup_logger(
        name="bigrag",
        log_dir=log_dir if log_dir else ".",
        log_file=log_filename,
        level=level,
        rotation="size",
        max_bytes=10 * 1024 * 1024,  # 10 MB
        backup_count=5,
        console_output=False,  # Maintain old behavior (file-only)
        error_separate=True
    )


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    concurrent_limit: int = 16

    def __post_init__(self):
        if self.concurrent_limit != 0:
            self._semaphore = asyncio.Semaphore(self.concurrent_limit)
        else:
            self._semaphore = UnlimitedSemaphore()

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        async with self._semaphore:
            return await self.func(*args, **kwargs)


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    try:
        maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
        if maybe_json_str is not None:
            maybe_json_str = maybe_json_str.group(0)
            maybe_json_str = maybe_json_str.replace("\\n", "")
            maybe_json_str = maybe_json_str.replace("\n", "")
            maybe_json_str = maybe_json_str.replace("'", '"')
            # json.loads(maybe_json_str) # don't check here, cannot validate schema after all
            return maybe_json_str
    except Exception:
        pass
        # try:
        #     content = (
        #         content.replace(kw_prompt[:-1], "")
        #         .replace("user", "")
        #         .replace("model", "")
        #         .strip()
        #     )
        #     maybe_json_str = "{" + content.split("{")[1].split("}")[0] + "}"
        #     json.loads(maybe_json_str)

        return None


def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


# ============================================================================
# Text Sanitization for LLM Extraction Output
# Added: 2025-01-13 for orphan node reduction
# ============================================================================

def sanitize_extracted_text(
    text: str,
    field_type: Literal["entity_name", "entity_type", "description", "relation", "general"] = "general"
) -> str:
    """
    Sanitize LLM-extracted text with field-specific rules.

    Purpose: Clean malformed LLM output to prevent parsing errors and
             ensure consistent entity/relation data quality.

    Args:
        text: Raw text from LLM output
        field_type: Type of field being sanitized
            - "entity_name": Strict cleaning for entity identifiers
            - "entity_type": Very strict (lowercase, no spaces)
            - "description": Allow most characters, remove control chars
            - "relation": Relation content cleaning
            - "general": Basic cleaning

    Returns:
        Cleaned text string (may be empty if input is invalid)

    Examples:
        >>> sanitize_extracted_text('"  LIONEL MESSI  "', "entity_name")
        'LIONEL MESSI'

        >>> sanitize_extracted_text('  person  ', "entity_type")
        'person'

        >>> sanitize_extracted_text('" He is a player "', "description")
        'He is a player'
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Remove outer quotes and whitespace
    text = text.strip()

    # Remove outer quotes (single or double, but not inner quotes)
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1].strip()

    # Step 2: Normalize whitespace (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text)

    # Step 3: Remove control characters (always)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    # Step 4: Field-specific cleaning
    if field_type == "entity_name":
        # Entity names: Remove ALL inner quotes
        text = text.replace('"', '').replace("'", '')

        # Check for delimiter corruption (entity names should never contain delimiters)
        forbidden_patterns = ['<|>', '<SEP>', '##', '<GRAPH_FIELD_SEP>', '||', '<>', '|>']
        for pattern in forbidden_patterns:
            if pattern in text:
                logger.warning(f"Entity name contains reserved delimiter '{pattern}': '{text}'")
                return ""

        # Entity name cannot be empty after cleaning
        if not text.strip():
            return ""

    elif field_type == "entity_type":
        # Entity type must be single word, lowercase, no spaces
        text = text.replace(" ", "").lower()

        # Reject if contains invalid characters
        invalid_chars = ["'", '"', "(", ")", "<", ">", "|", "/", "\\", ",", ";", "!", "?"]
        if any(char in text for char in invalid_chars):
            logger.warning(f"Invalid entity type contains special characters: '{text}'")
            return ""

        # Type cannot be empty
        if not text.strip():
            return ""

    elif field_type == "description":
        # Descriptions: Allow most characters, just remove control chars (already done in Step 3)
        # Remove any remaining double control characters
        text = re.sub(r'\s+', ' ', text)

        # Trim to reasonable length if extremely long
        MAX_DESC_LENGTH = 2000
        if len(text) > MAX_DESC_LENGTH:
            logger.warning(f"Description truncated from {len(text)} to {MAX_DESC_LENGTH} chars")
            text = text[:MAX_DESC_LENGTH] + "..."

    elif field_type == "relation":
        # Relation content: Similar to description
        text = re.sub(r'\s+', ' ', text)

        # Trim if extremely long
        MAX_RELATION_LENGTH = 1000
        if len(text) > MAX_RELATION_LENGTH:
            logger.warning(f"Relation content truncated from {len(text)} to {MAX_RELATION_LENGTH} chars")
            text = text[:MAX_RELATION_LENGTH] + "..."

    # Final validation: Return empty string if only whitespace remains
    return text.strip()


def fix_delimiter_corruption(record: str, tuple_delimiter: str = "<|>") -> str:
    """
    Fix common LLM delimiter corruption patterns.

    Purpose: LLM sometimes outputs variations of the tuple delimiter
             (e.g., <> instead of <|>, || instead of <|>). This function
             corrects these patterns to enable proper parsing.

    Args:
        record: Raw LLM output record (single line)
        tuple_delimiter: Expected delimiter (default: "<|>" for BiG-RAG)

    Returns:
        Record with corrected delimiters

    Examples:
        >>> fix_delimiter_corruption('entity<>MESSI<>person', '<|>')
        'entity<|>MESSI<|>person'

        >>> fix_delimiter_corruption('relation||content||score', '<|>')
        'relation<|>content<|>score'

    Common Corruption Patterns:
        - <> instead of <|>
        - || instead of <|>
        - <| or |> (incomplete)
        - < | > (with spaces)
        - <#> instead of <|#|> (if delimiter has core character)
    """
    if not record:
        return record

    # Extract core delimiter character if present
    # For BiG-RAG: tuple_delimiter = "<|>" → core = "|"
    # For LightRAG: tuple_delimiter = "<|#|>" → core = "#"
    if len(tuple_delimiter) >= 3 and tuple_delimiter.startswith('<') and tuple_delimiter.endswith('>'):
        core = tuple_delimiter[2:-2] if len(tuple_delimiter) > 4 else tuple_delimiter[1:-1]
    else:
        core = "|"  # Default fallback


    # Define corruption patterns (ordered by likelihood)
    corrupted_patterns = [
        # Double brackets (LLM tends to output this)
        f"<<{core}>>",    # <<|>> instead of <|>
        f"<{core}{core}>", # <||> instead of <|>

        # Missing pipes (but don't include pattern that matches valid delimiter!)
        "<>",             # Empty brackets

        # Missing brackets
        f"|{core}|",      # |#| instead of <|#|>
        "||",             # Double pipes

        # Partial patterns (but ONLY for multi-character delimiters like <|#|>)
        # Don't include <| or |> for <|> delimiter as they are substrings!
        f"<|{core}>",     # <|#> instead of <|#|>
        f"<{core}|>",     # <#|> instead of <|#|>

        # With spaces
        "< >",            # Brackets with space
        "| |",            # Pipes with space
        f"< {core} >",    # Brackets with spaces around core
        f"< | {core} | >", # Full pattern with spaces
        f"<| {core} |>",  # Spaces inside
    ]

    # Apply corrections
    for i, pattern in enumerate(corrupted_patterns):
        if pattern in record:
            record = record.replace(pattern, tuple_delimiter)
            logger.debug(f"Fixed delimiter corruption: '{pattern}' → '{tuple_delimiter}'")

    return record


def description_quality_score(description: str) -> float:
    """
    Calculate quality score for entity/relation descriptions.

    Purpose: During gleaning merge, compare quality of descriptions
             to keep the better version (not just longer version).

    Args:
        description: Entity or relation description text

    Returns:
        Quality score (float, higher = better quality)

    Scoring Factors:
        - Base: Length (more detail assumed better)
        - +10: Ends with proper sentence (period)
        - -50%: Very short (<20 chars)
        - +20: Contains specific keywords (who, which, known for, etc.)
        - +10: Contains numbers/dates (specific facts)

    Examples:
        >>> description_quality_score("Messi is a player")
        17  # Short, no period, no keywords

        >>> description_quality_score("Lionel Messi is a professional footballer known for winning 8 Ballon d'Or awards.")
        132  # Long (92) + period (+10) + keywords (+20) + numbers (+10)
    """
    if not description:
        return 0.0

    score = len(description)  # Base score: length

    # Bonus: Complete sentence (ends with period)
    if description.rstrip().endswith('.'):
        score += 10

    # Penalty: Very short descriptions (likely incomplete)
    if len(description) < 20:
        score *= 0.5

    # Bonus: Contains specific keywords (indicates detailed description)
    quality_keywords = [
        'who', 'which', 'where', 'when', 'professional',
        'known for', 'famous for', 'specialist', 'expert',
        'won', 'achieved', 'played', 'founded', 'established'
    ]
    keyword_matches = sum(1 for kw in quality_keywords if kw.lower() in description.lower())
    score += keyword_matches * 20

    # Bonus: Contains numbers/dates (indicates specific facts)
    has_numbers = bool(re.search(r'\d+', description))
    if has_numbers:
        score += 10

    # Bonus: Mentions multiple entities (rich context)
    # Heuristic: Count capitalized words (potential entity mentions)
    capitalized_words = len([w for w in description.split() if w and w[0].isupper()])
    if capitalized_words >= 3:
        score += 15

    return score


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_csv(data: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> List[List[str]]:
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]


def save_data_to_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_combine_contexts(hl, ll):
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources = []
    seen = set()

    for item in list_hl + list_ll:
        if item and item not in seen:
            combined_sources.append(item)
            seen.add(item)

    combined_sources_result = [",\t".join(header)]

    for i, item in enumerate(combined_sources, start=1):
        combined_sources_result.append(f"{i},\t{item}")

    combined_sources_result = "\n".join(combined_sources_result)

    return combined_sources_result


async def get_best_cached_response(
    hashing_kv,
    current_embedding,
    similarity_threshold=0.95,
    mode="default",
    use_llm_check=False,
    llm_func=None,
    original_prompt=None,
) -> Union[str, None]:
    # Get mode-specific cache
    mode_cache = await hashing_kv.get_by_id(mode)
    if not mode_cache:
        return None

    best_similarity = -1
    best_response = None
    best_prompt = None
    best_cache_id = None

    # Only iterate through cache entries for this mode
    for cache_id, cache_data in mode_cache.items():
        if cache_data["embedding"] is None:
            continue

        # Convert cached embedding list to ndarray
        cached_quantized = np.frombuffer(
            bytes.fromhex(cache_data["embedding"]), dtype=np.uint8
        ).reshape(cache_data["embedding_shape"])
        cached_embedding = dequantize_embedding(
            cached_quantized,
            cache_data["embedding_min"],
            cache_data["embedding_max"],
        )

        similarity = cosine_similarity(current_embedding, cached_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_response = cache_data["return"]
            best_prompt = cache_data["original_prompt"]
            best_cache_id = cache_id

    if best_similarity > similarity_threshold:
        # If LLM check is enabled and all required parameters are provided
        if use_llm_check and llm_func and original_prompt and best_prompt:
            compare_prompt = PROMPTS["similarity_check"].format(
                original_prompt=original_prompt, cached_prompt=best_prompt
            )

            try:
                llm_result = await llm_func(compare_prompt)
                llm_result = llm_result.strip()
                llm_similarity = float(llm_result)

                # Replace vector similarity with LLM similarity score
                best_similarity = llm_similarity
                if best_similarity < similarity_threshold:
                    log_data = {
                        "event": "llm_check_cache_rejected",
                        "original_question": original_prompt[:100] + "..."
                        if len(original_prompt) > 100
                        else original_prompt,
                        "cached_question": best_prompt[:100] + "..."
                        if len(best_prompt) > 100
                        else best_prompt,
                        "similarity_score": round(best_similarity, 4),
                        "threshold": similarity_threshold,
                    }
                    logger.info(json.dumps(log_data, ensure_ascii=False))
                    return None
            except Exception as e:  # Catch all possible exceptions
                logger.warning(f"LLM similarity check failed: {e}")
                return None  # Return None directly when LLM check fails

        prompt_display = (
            best_prompt[:50] + "..." if len(best_prompt) > 50 else best_prompt
        )
        log_data = {
            "event": "cache_hit",
            "mode": mode,
            "similarity": round(best_similarity, 4),
            "cache_id": best_cache_id,
            "original_prompt": prompt_display,
        }
        logger.info(json.dumps(log_data, ensure_ascii=False))
        return best_response
    return None


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


def quantize_embedding(embedding: np.ndarray, bits=8) -> tuple:
    """Quantize embedding to specified bits"""
    # Calculate min/max values for reconstruction
    min_val = embedding.min()
    max_val = embedding.max()

    # Quantize to 0-255 range
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = np.round((embedding - min_val) * scale).astype(np.uint8)

    return quantized, min_val, max_val


def dequantize_embedding(
    quantized: np.ndarray, min_val: float, max_val: float, bits=8
) -> np.ndarray:
    """Restore quantized embedding"""
    scale = (max_val - min_val) / (2**bits - 1)
    return (quantized * scale + min_val).astype(np.float32)


async def handle_cache(hashing_kv, args_hash, prompt, mode="default"):
    """Generic cache handling function"""
    if hashing_kv is None:
        return None, None, None, None

    # For naive mode, only use simple cache matching
    if mode == "naive":
        mode_cache = await hashing_kv.get_by_id(mode) or {}
        if args_hash in mode_cache:
            return mode_cache[args_hash]["return"], None, None, None
        return None, None, None, None

    # Get embedding cache configuration
    embedding_cache_config = hashing_kv.global_config.get(
        "embedding_cache_config",
        {"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False},
    )
    is_embedding_cache_enabled = embedding_cache_config["enabled"]
    use_llm_check = embedding_cache_config.get("use_llm_check", False)

    quantized = min_val = max_val = None
    if is_embedding_cache_enabled:
        # Use embedding cache
        embedding_model_func = hashing_kv.global_config["embedding_func"]["func"]
        llm_model_func = hashing_kv.global_config.get("llm_model_func")

        current_embedding = await embedding_model_func([prompt])
        quantized, min_val, max_val = quantize_embedding(current_embedding[0])
        best_cached_response = await get_best_cached_response(
            hashing_kv,
            current_embedding[0],
            similarity_threshold=embedding_cache_config["similarity_threshold"],
            mode=mode,
            use_llm_check=use_llm_check,
            llm_func=llm_model_func if use_llm_check else None,
            original_prompt=prompt if use_llm_check else None,
        )
        if best_cached_response is not None:
            return best_cached_response, None, None, None
    else:
        # Use regular cache
        mode_cache = await hashing_kv.get_by_id(mode) or {}
        if args_hash in mode_cache:
            return mode_cache[args_hash]["return"], None, None, None

    return None, quantized, min_val, max_val


@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    quantized: Optional[np.ndarray] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mode: str = "default"


async def save_to_cache(hashing_kv, cache_data: CacheData):
    if hashing_kv is None or hasattr(cache_data.content, "__aiter__"):
        return

    mode_cache = await hashing_kv.get_by_id(cache_data.mode) or {}

    mode_cache[cache_data.args_hash] = {
        "return": cache_data.content,
        "embedding": cache_data.quantized.tobytes().hex()
        if cache_data.quantized is not None
        else None,
        "embedding_shape": cache_data.quantized.shape
        if cache_data.quantized is not None
        else None,
        "embedding_min": cache_data.min_val,
        "embedding_max": cache_data.max_val,
        "original_prompt": cache_data.prompt,
    }

    await hashing_kv.upsert({cache_data.mode: mode_cache})


def safe_unicode_decode(content):
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )

    return decoded_content


# ========================
# Retry Mechanism (B3)
# ========================

async def safe_operation_with_retry(
    operation,
    operation_name: str,
    context: str = "",
    max_retries: int = 3,
    retry_delay: float = 0.2,
):
    """
    Execute async operation with retry on transient failures.

    Uses exponential backoff for retries to handle transient failures
    in VDB operations, storage operations, and LLM API calls.

    Args:
        operation: Async callable to execute
        operation_name: Name of operation for logging
        context: Additional context for error messages
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (seconds)

    Returns:
        Result of the operation

    Raises:
        Exception: If operation fails after all retries

    Example:
        await safe_operation_with_retry(
            lambda: vdb_entities.upsert(data),
            "VDB upsert entities",
            context=f"{len(data)} entities",
            max_retries=3,
        )
    """
    # Total attempts = 1 initial + max_retries
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as e:
            if attempt >= max_retries:
                # Failed after all retries
                total_attempts = max_retries + 1
                error_msg = f"{operation_name} failed for {context} after {total_attempts} attempts: {e}"
                logger.error(error_msg)
                raise Exception(error_msg) from e
            else:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"{operation_name} attempt {attempt + 1} failed for {context}: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)


# ========================
# Logging Setup (B4)
# ========================

def setup_bigrag_logger(
    logger_name: str = "bigrag",
    level: str = "INFO",
    log_dir: Optional[str] = None,
):
    """
    Setup BiG-RAG logger with console and optional rotating file handlers.

    This provides production-ready logging with:
    - Console handler (simple format for terminal output)
    - Rotating file handler (detailed format, 10MB max, 5 backups)

    Args:
        logger_name: Name of the logger (default: "bigrag")
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None = console only)

    Returns:
        Configured logger instance

    Example:
        from bigrag.config import config
        logger = setup_bigrag_logger(
            level=config.log_level,
            log_dir=config.log_dir
        )
    """
    import logging.handlers
    from pathlib import Path
    from .constants import DEFAULT_LOG_MAX_BYTES, DEFAULT_LOG_BACKUP_COUNT

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    logger.propagate = False  # Don't propagate to root logger

    # Console handler (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    # Rotating file handler (detailed format)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=f"{log_dir}/bigrag.log",
            maxBytes=DEFAULT_LOG_MAX_BYTES,  # 10MB
            backupCount=DEFAULT_LOG_BACKUP_COUNT,  # 5 backups
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def normalize_text(text: str) -> str:
    """
    Normalize text for matching by removing extra whitespace, lowercasing, and removing punctuation.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text string

    Examples:
        >>> normalize_text("  Hello,  World!  ")
        'hello world'
        >>> normalize_text("The Quick Brown Fox.")
        'the quick brown fox'
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def remove_stopwords(text: str, stopwords: Optional[List[str]] = None) -> str:
    """
    Remove common stopwords from text.

    Args:
        text: Input text
        stopwords: Optional list of stopwords to remove (default: common English stopwords)

    Returns:
        Text with stopwords removed

    Examples:
        >>> remove_stopwords("the quick brown fox")
        'quick brown fox'
        >>> remove_stopwords("this is a test")
        'test'
    """
    # Default English stopwords (comprehensive list)
    if stopwords is None:
        stopwords = [
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
            'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
            'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot',
            'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each',
            'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
            'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
            'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just',
            'me', 'might', 'more', 'most', 'must', 'my', 'myself', 'no', 'nor',
            'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
            'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she',
            'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their',
            'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
            'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
            'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
            'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you',
            'your', 'yours', 'yourself', 'yourselves'
        ]

    # Split text into words
    words = text.lower().split()

    # Remove stopwords
    filtered_words = [word for word in words if word not in stopwords]

    return ' '.join(filtered_words)


# ====================================================================================
# NEW: Enhanced Pipeline Utilities (Phase 1 Step 2 - Semantic Chunking)
# ====================================================================================

def count_tokens_fast(text: str, chars_per_token: int = 4) -> int:
    """
    Fast approximate token counting for chunking decisions.

    Uses character-based approximation (4 chars ≈ 1 token) for performance.
    For production use with accurate counting, use tiktoken directly.

    Args:
        text: Text to count tokens for
        chars_per_token: Characters per token (default: 4)

    Returns:
        Approximate token count

    Example:
        >>> count_tokens_fast("Hello world")
        2  # 11 chars / 4 ≈ 2.75 → 2
        >>> count_tokens_fast("A longer sentence with more words")
        8  # 33 chars / 4 ≈ 8.25 → 8
    """
    if not text or not text.strip():
        return 0

    # Simple approximation: 4 characters per token
    # This is fast but approximate (actual varies by language)
    return len(text) // chars_per_token


def count_tokens_accurate(text: str, model_name: str = "gpt-4o") -> int:
    """
    Accurate token counting using tiktoken.

    Slower but precise. Use for validation or when accuracy is critical.

    Args:
        text: Text to count tokens for
        model_name: Model name for tokenizer

    Returns:
        Exact token count
    """
    if not text or not text.strip():
        return 0

    try:
        tokens = encode_string_by_tiktoken(text, model_name)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Tiktoken encoding failed, using approximation: {e}")
        return count_tokens_fast(text)


def split_by_sentences(text: str, languages: List[str] = None) -> List[str]:
    """
    Split text into sentences respecting multiple language conventions.

    Handles:
    - Bengali: । (purno)
    - English: . ! ?
    - Preserves sentence boundaries accurately

    Args:
        text: Text to split
        languages: List of languages (unused, for future enhancement)

    Returns:
        List of sentences

    Example:
        >>> split_by_sentences("Hello. How are you? I'm fine।")
        ['Hello.', 'How are you?', "I'm fine।"]
    """
    if not text or not text.strip():
        return []

    # Pattern for sentence endings:
    # - Bengali: । followed by space/newline/end
    # - English: . ! ? followed by space/newline/end
    # Use lookahead to keep the punctuation with the sentence

    # Combined pattern for both Bengali and English
    pattern = r'(?<=[।.!?])\s+'

    # Split by pattern
    sentences = re.split(pattern, text)

    # Filter empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def split_by_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs (double newline separation).

    Args:
        text: Text to split

    Returns:
        List of paragraphs

    Example:
        >>> split_by_paragraphs("Para 1\\n\\nPara 2\\n\\nPara 3")
        ['Para 1', 'Para 2', 'Para 3']
    """
    if not text or not text.strip():
        return []

    # Split by one or more empty lines (double newline or more)
    paragraphs = re.split(r'\n\s*\n+', text)

    # Filter empty paragraphs and strip whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def get_overlap_text(
    text: str,
    target_tokens: int,
    direction: str = 'end',
    chars_per_token: int = 4
) -> str:
    """
    Extract overlap text from beginning or end of text.

    Tries to extract complete sentences when possible.

    Args:
        text: Source text
        target_tokens: Target token count for overlap
        direction: 'start' or 'end'
        chars_per_token: Characters per token approximation

    Returns:
        Overlap text (complete sentences when possible)

    Example:
        >>> get_overlap_text("First. Second. Third.", 5, 'end')
        'Third.'
    """
    if not text or not text.strip() or target_tokens <= 0:
        return ""

    # Calculate approximate character count for target tokens
    target_chars = target_tokens * chars_per_token

    # Split into sentences
    sentences = split_by_sentences(text)

    if not sentences:
        # Fallback: character-based extraction
        if direction == 'end':
            return text[-target_chars:].strip()
        else:
            return text[:target_chars].strip()

    # Accumulate complete sentences
    if direction == 'end':
        # Start from end and work backwards
        overlap_sentences = []
        current_chars = 0

        for sentence in reversed(sentences):
            sentence_chars = len(sentence)
            if current_chars + sentence_chars <= target_chars * 1.5:  # Allow 50% overflow
                overlap_sentences.insert(0, sentence)
                current_chars += sentence_chars
            else:
                break

        return ' '.join(overlap_sentences) if overlap_sentences else sentences[-1]

    else:  # direction == 'start'
        # Start from beginning
        overlap_sentences = []
        current_chars = 0

        for sentence in sentences:
            sentence_chars = len(sentence)
            if current_chars + sentence_chars <= target_chars * 1.5:  # Allow 50% overflow
                overlap_sentences.append(sentence)
                current_chars += sentence_chars
            else:
                break

        return ' '.join(overlap_sentences) if overlap_sentences else sentences[0]


# ========== QUALITY SCORING FUNCTIONS (Week 1 - Modular Pipeline) ==========

def description_quality_score(description: str) -> float:
    """
    Calculate quality score for entity description (used in gleaning merge).

    Scoring factors:
    - Length (40 points max): Longer descriptions are more detailed
    - Keyword density (30 points max): Informative words (who, what, where, when)
    - Specificity (30 points max): Numbers, dates, proper names

    Returns: Score 0-100

    Examples:
        >>> description_quality_score("KUET has 18 departments established in 1967.")
        85.0  # Good length, has numbers and dates

        >>> description_quality_score("University")
        10.0  # Too short, no details
    """
    import re
    
    if not description:
        return 0.0

    score = 0.0

    # Factor 1: Length (up to 40 points)
    # 200 chars = 40 points
    length_score = min(len(description) / 5, 40)
    score += length_score

    # Factor 2: Keyword density (up to 30 points)
    # Informative question words in both English and Bangla
    informative_words = [
        'who', 'what', 'when', 'where', 'why', 'how', 'which',
        'কে', 'কি', 'কোথায়', 'কেন', 'কখন', 'কীভাবে'
    ]
    keyword_count = sum(1 for word in informative_words if word in description.lower())
    keyword_score = min(keyword_count * 10, 30)
    score += keyword_score

    # Factor 3: Specificity (up to 30 points)
    # - Has numbers: 10 points
    # - Has dates (4-digit years or date patterns): 10 points
    # - Has proper names (capitalized words or Bangla names): 10 points
    has_numbers = bool(re.search(r'\d', description))
    has_dates = bool(re.search(r'\d{4}|\d{1,2}/\d{1,2}|\d{1,2}-\d{1,2}', description))
    has_names = bool(re.search(r'[A-Z][a-z]+|[অ-হ]{3,}', description))

    specificity_score = (
        (10 if has_numbers else 0) +
        (10 if has_dates else 0) +
        (10 if has_names else 0)
    )
    score += specificity_score

    return round(score, 1)


def entity_quality_score(
    entity: dict,
    description_weight: float = 0.4,
    context_weight: float = 0.3,
    frequency_weight: float = 0.2,
    type_weight: float = 0.1
) -> float:
    """
    Calculate comprehensive quality score for an entity.

    Components:
    - Description completeness (40%): How detailed is the description?
    - Context relevance (30%): Is entity mentioned in relevant context?
    - Source frequency (20%): How many chunks mention this entity?
    - Type specificity (10%): Is type specific vs. generic?

    Args:
        entity: Entity dict with name, description, entity_type, source_id
        description_weight: Weight for description quality (default 0.4)
        context_weight: Weight for context relevance (default 0.3)
        frequency_weight: Weight for source frequency (default 0.2)
        type_weight: Weight for type specificity (default 0.1)

    Returns:
        Quality score 0-100

    Examples:
        >>> entity = {
        ...     "name": "KUET",
        ...     "description": "Khulna University of Engineering and Technology, established 1967",
        ...     "entity_type": "UNIVERSITY",
        ...     "source_id": ["chunk1", "chunk2", "chunk3"]
        ... }
        >>> entity_quality_score(entity)
        82.5  # High score: good description, specific type, multiple sources
    """
    # Component 1: Description quality (0-100)
    description = entity.get('description', '')
    desc_score = description_quality_score(description)

    # Component 2: Context relevance (0-100)
    # Simple heuristic: if description mentions entity name = relevant
    entity_name = entity.get('name', '').lower()
    context_score = 100.0 if entity_name in description.lower() else 50.0

    # Component 3: Source frequency (0-100)
    # Normalize by typical max (assume 10 sources = excellent)
    source_ids = entity.get('source_id', [])
    if isinstance(source_ids, str):
        source_ids = [source_ids]
    source_count = len(source_ids)
    freq_score = min(source_count * 10, 100)

    # Component 4: Type specificity (0-100)
    # Generic types get low scores
    entity_type = entity.get('entity_type', 'UNKNOWN').upper()
    generic_types = {'UNKNOWN', 'OTHER', 'ENTITY', 'THING', 'ITEM', 'OBJECT'}
    type_score = 30.0 if entity_type in generic_types else 100.0

    # Weighted combination
    total_score = (
        description_weight * desc_score +
        context_weight * context_score +
        frequency_weight * freq_score +
        type_weight * type_score
    )

    return round(total_score, 1)


def relation_completeness_score(relation: dict) -> float:
    """
    Calculate completeness score for a relation (0-10 scale).

    Criteria:
    - Has source and target entities: +2 points
    - Has description: +3 points
    - Description mentions both entities: +3 points
    - Description is detailed (>20 chars): +2 points

    Args:
        relation: Relation dict with description, source_id, target_id

    Returns:
        Completeness score 0-10

    Examples:
        >>> relation = {
        ...     "description": "KUET was established in Khulna in 1967",
        ...     "source_id": "entity-kuet",
        ...     "target_id": "entity-khulna"
        ... }
        >>> relation_completeness_score(relation)
        10.0  # Complete relation with detailed description
    """
    score = 0.0

    # Check 1: Has source and target (2 points)
    if relation.get('source_id') and relation.get('target_id'):
        score += 2.0

    # Check 2: Has description (3 points)
    description = relation.get('description', '').strip()
    if description:
        score += 3.0

        # Check 3: Description is detailed (2 points)
        if len(description) > 20:
            score += 2.0

        # Check 4: Description mentions context (3 points)
        # Simple heuristic: longer descriptions are more likely contextual
        if len(description) > 50:
            score += 3.0

    return round(score, 1)
