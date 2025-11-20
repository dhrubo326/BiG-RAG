"""
Entity merging for BiG-RAG production knowledge graph construction.

This module contains entity canonicalization and linking components.
"""

from bigrag.merging.canonicalization import EntityCanonicalizationMap
from bigrag.merging.entity_linker import ProductionEntityLinker

__all__ = ['EntityCanonicalizationMap', 'ProductionEntityLinker']
