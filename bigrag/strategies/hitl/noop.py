from bigrag.interfaces.hitl import HITLInterface
from typing import List, Dict, Optional

class NoOpHITL(HITLInterface):
    async def save_failures(self, failed_chunks: List[Dict], metadata: Optional[Dict] = None) -> None:
        pass

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
        """No-op implementation - does nothing."""
        pass
