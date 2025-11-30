"""
ValidatorInterface - Abstract interface for extraction validation strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict


class ValidatorInterface(ABC):
    """Abstract interface for extraction validation strategies."""

    @abstractmethod
    async def validate(self, extractions: Dict) -> Dict:
        """
        Validate extractions (numeric coverage, semantic quality, etc.).

        Args:
            extractions: Output from ExtractorInterface.extract()

        Returns:
            {
                'entities': [...],       # Valid entities
                'relations': [...],      # Valid relations
                'failed_chunks': [...],  # Chunks that failed validation
                'summary': {
                    'status': 'PASS' | 'WARNING' | 'FAIL',
                    'numeric_coverage': 0.95,
                    'semantic_validity': 0.98
                }
            }
        """
        pass
