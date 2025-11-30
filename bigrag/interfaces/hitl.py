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
