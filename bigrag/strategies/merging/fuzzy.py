from bigrag.interfaces.merger import MergerInterface
from typing import List, Dict

class FuzzyMerger(MergerInterface):
    def __init__(self, fuzzy_threshold: float = 0.90):
        self.fuzzy_threshold = fuzzy_threshold
        try:
            from bigrag.merging.entity_linker import SimpleEntityLinker
            from bigrag.merging.canonicalization import EntityCanonicalizationMap
            self.entity_linker = SimpleEntityLinker(EntityCanonicalizationMap())
        except:
            self.entity_linker = None

    async def merge(self, entities: List[Dict]) -> List[Dict]:
        if self.entity_linker:
            try:
                return await self.entity_linker.link_entities_across_chunks(entities)
            except Exception as e:
                print(f"[WARNING] Fuzzy merging failed: {e}. Falling back to basic merge.")
        from bigrag.strategies.merging.basic import BasicMerger
        return await BasicMerger().merge(entities)
