from bigrag.interfaces.orphan_linker import OrphanLinkerInterface
from typing import List, Dict, Tuple

class NoOpOrphanLinker(OrphanLinkerInterface):
    async def link(self, entities: List[Dict], relations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        return (entities, [])
