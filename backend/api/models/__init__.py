"""
Pydantic models for BiG-RAG API

Request and response models organized by domain.

For backward compatibility, all models are re-exported from this module.
"""

# Re-export all models for backward compatibility
try:
    from .models import *
except ImportError:
    # Fallback to parent directory if models not copied yet
    import sys
    from pathlib import Path
    parent = Path(__file__).parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from models import *

try:
    from .models_eval import *
except ImportError:
    # Fallback to parent directory
    from models_eval import *

__version__ = "1.0.0"
