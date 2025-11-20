"""
Preprocessors for BiG-RAG production knowledge graph construction.

This module contains pre-extraction processing components for educational domain.
"""

from bigrag.preprocessors.table_extractor import GPT4TableExtractor, BilingualDetector

__all__ = ['GPT4TableExtractor', 'BilingualDetector']
