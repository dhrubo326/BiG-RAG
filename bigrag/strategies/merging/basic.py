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

            # CRITICAL: Collect all entity_ids that were merged (required for entity ID remapping in Step 5.5)
            # Without this, relations may reference old entity_ids that no longer exist after merge
            m['entity_ids_merged'] = [e.get('entity_id') for e in group if e.get('entity_id')]

            merged.append(m)
        return merged
