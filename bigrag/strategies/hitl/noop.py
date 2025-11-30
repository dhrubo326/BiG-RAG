from bigrag.interfaces.hitl import HITLInterface
from typing import List, Dict, Optional

class NoOpHITL(HITLInterface):
    async def save_failures(self, failed_chunks: List[Dict], metadata: Optional[Dict] = None) -> None:
        pass
