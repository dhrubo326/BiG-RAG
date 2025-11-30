from bigrag.interfaces.orphan_linker import OrphanLinkerInterface
from typing import List, Dict, Tuple
import hashlib

class SyntheticOrphanLinker(OrphanLinkerInterface):
    async def link(self, entities: List[Dict], relations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        orphans = [e for e in entities if not e.get('hyper_relation')]
        if not orphans:
            return (entities, [])
        synthetic_relations = []
        for orphan in orphans:
            rel_id = hashlib.md5(f"synthetic_{orphan.get('entity_name', '')}".encode()).hexdigest()[:16]
            synthetic_rel = {
                'relation_name': f"mentioned_{orphan.get('entity_type', 'entity')}",
                'description': f"Synthetic relation for {orphan.get('entity_name', '')}",
                'weight': orphan.get('weight', 0) * 0.5,
                'is_synthetic': True
            }
            synthetic_relations.append(synthetic_rel)
            orphan['hyper_relation'] = f'rel-{rel_id}'
        return (entities, synthetic_relations)
