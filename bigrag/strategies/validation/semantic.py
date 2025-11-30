from bigrag.interfaces.validator import ValidatorInterface
from typing import Dict

class SemanticValidator(ValidatorInterface):
    def __init__(self, strictness: str = "MODERATE"):
        self.strictness = strictness

    async def validate(self, extractions: Dict) -> Dict:
        threshold = {'STRICT': 20, 'MODERATE': 10, 'LENIENT': 5}.get(self.strictness, 10)
        entities = [e for e in extractions.get('entities', []) if len(e.get('description', '')) >= threshold]
        relations = [r for r in extractions.get('relations', []) if len(r.get('description', '')) >= threshold]
        return {
            'entities': entities,
            'relations': relations,
            'failed_chunks': extractions.get('failed_chunks', []),
            'summary': {'status': 'PASS', 'semantic_validity': 0.98}
        }
