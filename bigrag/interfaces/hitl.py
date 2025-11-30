"""
HITLInterface - Abstract interface for Human-in-the-Loop strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class HITLInterface(ABC):
    """Abstract interface for Human-in-the-Loop strategies."""

    @abstractmethod
    async def save_failures(
        self,
        failed_chunks: List[Dict],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save failed extractions for human review.

        Args:
            failed_chunks: Chunks that failed validation
            metadata: Optional document metadata
        """
        pass

    @abstractmethod
    async def save_failed_table(
        self,
        chunk_id: str,
        table_id: str,
        reason: str,
        validation_feedback: str = '',
        missing_numbers: Optional[List] = None,
        hallucinated_numbers: Optional[List] = None,
        numeric_coverage: float = 0.0,
        source_markdown: str = '',
        extracted_data: Optional[Dict] = None,
        error_traceback: Optional[str] = None
    ) -> None:
        """
        Save failed table extraction with rich validation metadata.

        Args:
            chunk_id: Chunk identifier
            table_id: Table identifier
            reason: Failure reason
            validation_feedback: LLM validation feedback
            missing_numbers: Numbers in source but not in extraction
            hallucinated_numbers: Numbers in extraction but not in source
            numeric_coverage: Percentage of numbers correctly extracted
            source_markdown: Original table markdown
            extracted_data: Extracted structured data
            error_traceback: Exception traceback (if extraction error)
        """
        pass
