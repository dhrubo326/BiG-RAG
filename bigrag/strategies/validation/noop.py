from bigrag.interfaces.validator import ValidatorInterface
from typing import Dict

class NoOpValidator(ValidatorInterface):
    async def validate(self, extractions: Dict) -> Dict:
        return {
            'entities': extractions.get('entities', []),
            'relations': extractions.get('relations', []),
            'failed_chunks': [],
            'summary': {'status': 'PASS', 'numeric_coverage': 1.0}
        }
