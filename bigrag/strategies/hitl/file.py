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
