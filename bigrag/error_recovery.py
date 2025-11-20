"""
Production-Grade Error Handling and Recovery

This module provides retry logic with exponential backoff and checkpointing
for long-running operations like knowledge graph construction.

Critical for:
- API rate limit handling (GPT-4o, embedding models)
- Network timeout recovery
- Resume capability for 1000+ document processing
"""

import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Callable, Any, Dict
from functools import wraps

logger = logging.getLogger(__name__)


class ExtractionErrorHandler:
    """
    Production-grade error handling with automatic retries.

    ENHANCED with exponential backoff and specific error handling.
    """

    @staticmethod
    async def retry_with_backoff(
        async_func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        on_error: Callable = None
    ) -> Any:
        """
        Exponential backoff retry for API calls.

        Retry delays: 1s, 2s, 4s (exponential)

        Examples:
            >>> async def call_gpt4():
            ...     return await openai_client.chat.completions.create(...)
            >>>
            >>> result = await ExtractionErrorHandler.retry_with_backoff(call_gpt4)

        Args:
            async_func: Async function to retry
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds (default: 1.0)
            max_delay: Maximum delay in seconds (default: 10.0)
            on_error: Optional callback for error logging

        Returns:
            Result from successful function execution

        Raises:
            Exception: If all retries fail or error is non-retryable
        """
        for attempt in range(max_retries):
            try:
                return await async_func()
            except Exception as e:
                # Check if retryable error
                is_retryable = ExtractionErrorHandler._is_retryable_error(e)

                if attempt == max_retries - 1 or not is_retryable:
                    if on_error:
                        on_error(e)

                    # Log final failure
                    logger.error(
                        f"Function {async_func.__name__ if hasattr(async_func, '__name__') else 'anonymous'} "
                        f"failed after {attempt + 1} attempts: {e}"
                    )
                    raise

                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """
        Determine if error is retryable.

        Retryable errors:
        - Network timeouts
        - API rate limits (429)
        - Temporary server errors (500, 502, 503)

        Non-retryable errors:
        - Invalid API key (401)
        - Malformed request (400)
        - Not found (404)

        Args:
            error: Exception to check

        Returns:
            True if error should be retried, False otherwise
        """
        error_str = str(error).lower()

        # Retryable patterns
        retryable_patterns = [
            'timeout', 'timed out',
            'rate limit', '429',
            'server error', '500', '502', '503',
            'connection error', 'connection reset',
            'service unavailable', 'bad gateway'
        ]

        # Non-retryable patterns
        non_retryable_patterns = [
            'invalid api key', '401', 'unauthorized',
            'bad request', '400',
            'not found', '404',
            'forbidden', '403'
        ]

        # Check non-retryable first (fail fast)
        if any(pattern in error_str for pattern in non_retryable_patterns):
            return False

        # Check retryable
        if any(pattern in error_str for pattern in retryable_patterns):
            return True

        # Default: retry for unknown errors (conservative approach)
        return True

    @staticmethod
    def create_checkpoint(
        document_id: str,
        phase: str,
        data: Dict
    ):
        """
        Create checkpoint after each document processing.

        Enables resume from failure (critical for large batches).

        Examples:
            >>> ExtractionErrorHandler.create_checkpoint(
            ...     document_id="doc_001",
            ...     phase="table_extraction",
            ...     data={"tables": 5, "chunks": 12}
            ... )

        Args:
            document_id: Unique document identifier
            phase: Processing phase (e.g., "table_extraction", "entity_linking")
            data: Data to checkpoint
        """
        checkpoint_dir = Path("expr/checkpoints")
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        checkpoint_file = checkpoint_dir / f"{document_id}_{phase}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                'document_id': document_id,
                'phase': phase,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }, f, ensure_ascii=False, indent=2)

        logger.debug(f"Checkpoint created: {checkpoint_file}")

    @staticmethod
    def load_checkpoint(
        document_id: str,
        phase: str
    ) -> Dict | None:
        """
        Load checkpoint if exists.

        Examples:
            >>> checkpoint = ExtractionErrorHandler.load_checkpoint(
            ...     document_id="doc_001",
            ...     phase="table_extraction"
            ... )
            >>> if checkpoint:
            ...     print(f"Resuming from: {checkpoint['timestamp']}")

        Args:
            document_id: Unique document identifier
            phase: Processing phase

        Returns:
            Checkpoint data if exists, None otherwise
        """
        checkpoint_file = Path(f"expr/checkpoints/{document_id}_{phase}.json")
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    @staticmethod
    def clear_checkpoints(document_id: str = None):
        """
        Clear checkpoints (after successful completion).

        Args:
            document_id: If provided, clear only this document's checkpoints.
                        If None, clear all checkpoints.
        """
        checkpoint_dir = Path("expr/checkpoints")
        if not checkpoint_dir.exists():
            return

        if document_id:
            # Clear specific document checkpoints
            for checkpoint_file in checkpoint_dir.glob(f"{document_id}_*.json"):
                checkpoint_file.unlink()
                logger.debug(f"Cleared checkpoint: {checkpoint_file}")
        else:
            # Clear all checkpoints
            for checkpoint_file in checkpoint_dir.glob("*.json"):
                checkpoint_file.unlink()
            logger.info("Cleared all checkpoints")


def retry_on_failure(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for automatic retry with exponential backoff.

    Examples:
        >>> @retry_on_failure(max_retries=3, base_delay=2.0)
        ... async def my_llm_call(prompt: str):
        ...     return await openai_client.chat.completions.create(
        ...         model="gpt-4o",
        ...         messages=[{"role": "user", "content": prompt}]
        ...     )
        >>>
        >>> result = await my_llm_call("Extract entities from this text...")

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds

    Returns:
        Decorated async function with retry logic
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def _func():
                return await func(*args, **kwargs)

            return await ExtractionErrorHandler.retry_with_backoff(
                _func,
                max_retries=max_retries,
                base_delay=base_delay
            )
        return wrapper
    return decorator


# Usage Examples for documentation
if __name__ == "__main__":
    print("Error Recovery Module - Usage Examples")
    print("=" * 50)

    print("\n1. Basic retry with backoff:")
    print("""
    async def risky_operation():
        # May fail with rate limit or timeout
        result = await api_call()
        return result

    result = await ExtractionErrorHandler.retry_with_backoff(risky_operation)
    """)

    print("\n2. Using decorator:")
    print("""
    @retry_on_failure(max_retries=3, base_delay=2.0)
    async def extract_table(table_text: str):
        return await gpt4_table_extractor.extract(table_text)

    tables = await extract_table(markdown_table)
    """)

    print("\n3. Checkpointing:")
    print("""
    # Save progress
    ExtractionErrorHandler.create_checkpoint(
        document_id="doc_001",
        phase="table_extraction",
        data={"completed": 5, "total": 10}
    )

    # Resume later
    checkpoint = ExtractionErrorHandler.load_checkpoint("doc_001", "table_extraction")
    if checkpoint:
        start_from = checkpoint['data']['completed']
    """)

    print("\n" + "=" * 50)
    print("Error Recovery Module is ready for use!")
