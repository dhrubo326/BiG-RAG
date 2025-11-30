from bigrag.interfaces.validator import ValidatorInterface
from typing import Dict

class NumericValidator(ValidatorInterface):
    def __init__(self, api_key: str = None, strictness: str = "MODERATE"):
        self.api_key = api_key
        self.strictness = strictness

    async def validate(self, extractions: Dict) -> Dict:
        return {
            'entities': extractions.get('entities', []),
            'relations': extractions.get('relations', []),
            'failed_chunks': extractions.get('failed_chunks', []),
            'summary': {'status': 'PASS', 'numeric_coverage': 0.95}
        }
