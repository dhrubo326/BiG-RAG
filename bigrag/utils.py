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
from typing import Any, Union, List, Optional
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


def set_logger(log_file: str, level: str = "INFO"):
    """
    Initialize logger with file handler (backward compatibility wrapper)

    This function now uses the centralized logging_config module
    for better log rotation and management.

    Args:
        log_file: Path to log file
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
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
