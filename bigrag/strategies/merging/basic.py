from bigrag.interfaces.merger import MergerInterface
from typing import List, Dict
from collections import defaultdict

class BasicMerger(MergerInterface):
    async def merge(self, entities: List[Dict]) -> List[Dict]:
        groups = defaultdict(list)
        for e in entities:
            name = e.get('entity_name', '').lower().strip()
            if name:
                groups[name].append(e)
        merged = []
        for name, group in groups.items():
            m = group[0].copy()
            m['weight'] = sum(e.get('weight', 0) for e in group)
            m['source_id'] = list(set(
                sid for e in group
                for sid in (e.get('source_id') if isinstance(e.get('source_id'), list) else [e.get('source_id')])
                if sid
            ))
            merged.append(m)
        return merged
