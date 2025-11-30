from bigrag.interfaces.validator import ValidatorInterface
from typing import List, Dict

class CompositeValidator(ValidatorInterface):
    def __init__(self, validators: List[ValidatorInterface]):
        self.validators = validators

    async def validate(self, extractions: Dict) -> Dict:
        result = extractions
        summaries = []
        for v in self.validators:
            result = await v.validate(result)
            summaries.append(result['summary'])
        statuses = [s.get('status', 'PASS') for s in summaries]
        status = 'FAIL' if 'FAIL' in statuses else ('WARNING' if 'WARNING' in statuses else 'PASS')
        result['summary'] = {
            'status': status,
            'validators_run': [type(v).__name__ for v in self.validators],
            'individual_summaries': summaries
        }
        return result
