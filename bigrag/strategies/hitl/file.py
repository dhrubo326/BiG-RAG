from bigrag.interfaces.hitl import HITLInterface
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime

class FileHITL(HITLInterface):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    async def save_failures(self, failed_chunks: List[Dict], metadata: Optional[Dict] = None) -> None:
        if not failed_chunks:
            return
        failed_dir = Path(self.dataset_path) / 'failed_extractions'
        failed_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = failed_dir / f'failed_chunks_{ts}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': ts,
                'metadata': metadata or {},
                'failed_chunks': failed_chunks,
                'count': len(failed_chunks)
            }, f, indent=2)
        print(f'[HITL] Saved {len(failed_chunks)} failed chunks to {output_file}')

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
        Save failed table with rich validation metadata.

        COPIED FROM production_pipeline.py:161-171
        Preserves rich validation metadata for failed tables (for human review).
        """
        failed_dir = Path(self.dataset_path) / 'failed_extractions'
        failed_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = failed_dir / f'failed_table_{chunk_id}_{ts}.json'

        # Build rich metadata matching old production_pipeline.py structure
        failure_record = {
            'chunk_id': chunk_id,
            'table_id': table_id,
            'reason': reason,
            'validation_feedback': validation_feedback,
            'missing_numbers': missing_numbers or [],
            'hallucinated_numbers': hallucinated_numbers or [],
            'numeric_coverage': numeric_coverage,
            'source_markdown': source_markdown,
            'extracted_data': extracted_data or {},
            'timestamp': ts
        }

        # Add error traceback if present (for extraction errors)
        if error_traceback:
            failure_record['error_traceback'] = error_traceback

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(failure_record, f, indent=2, ensure_ascii=False)

        print(f'[HITL] Saved failed table {chunk_id} (reason: {reason}) to {output_file}')
