"""
Human-in-the-Loop (HITL) System

Provides infrastructure for capturing, storing, and managing failed extractions
for human review and correction.

Components:
- FailedExtractionStore: Storage and retrieval of failed extractions
- Review queue management
- Correction submission and reprocessing

Part of Phase 1 Step 6: HITL System
"""

from .failed_extraction_store import FailedExtractionStore

__all__ = ['FailedExtractionStore']
