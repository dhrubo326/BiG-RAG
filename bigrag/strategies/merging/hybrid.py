from bigrag.interfaces.merger import MergerInterface
from typing import List, Dict

class HybridMerger(MergerInterface):
    async def merge(self, entities: List[Dict]) -> List[Dict]:
        if len(entities) > 1000:
            from bigrag.strategies.merging.basic import BasicMerger
            return await BasicMerger().merge(entities)
        from bigrag.strategies.merging.fuzzy import FuzzyMerger
        return await FuzzyMerger().merge(entities)
