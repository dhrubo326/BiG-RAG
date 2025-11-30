"""
Language Detection Utility for BiG-RAG

Provides automatic language detection for document indexing with cascading fallback logic.
Uses BilingualDetector from table_extractor module for language detection.
"""

from typing import Optional
import os


# Language code to full name mapping
LANGUAGE_CODE_MAP = {
    'bn': 'Bangla',
    'en': 'English',
    'hi': 'Hindi',
    'ar': 'Arabic',
    'zh': 'Chinese',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ja': 'Japanese',
    'ko': 'Korean'
}


def detect_document_language(text: str) -> Optional[str]:
    """
    Auto-detect primary language from document content.

    Uses BilingualDetector to analyze text and determine primary language.
    Returns full language name (e.g., "Bangla", "English") instead of code.

    Args:
        text: Document content to analyze

    Returns:
        Full language name (e.g., "Bangla") or None if detection fails

    Examples:
        >>> detect_document_language("This is English text")
        'English'
        >>> detect_document_language("এটি বাংলা টেক্সট")
        'Bangla'
        >>> detect_document_language("Mixed বাংলা and English")
        'Bangla'  # Returns primary language
    """
    try:
        from bigrag.preprocessors.table_extractor import BilingualDetector

        # Detect language using BilingualDetector
        lang_info = BilingualDetector.detect_languages(text)

        # Extract primary language code
        primary_code = lang_info.get('primary')

        if not primary_code:
            return None

        # Map code to full name
        return LANGUAGE_CODE_MAP.get(primary_code, None)

    except Exception:
        # If detection fails for any reason, return None
        # (will fall back to .env or hardcoded default)
        return None


def get_language_with_fallback(
    explicit_language: Optional[str] = None,
    document_text: Optional[str] = None,
    env_default: bool = True
) -> str:
    """
    Get language using cascading fallback logic.

    Priority:
    1. explicit_language (user-specified via API parameter)
    2. Auto-detected from document_text
    3. DEFAULT_LANGUAGE from .env file (if env_default=True)
    4. Hardcoded fallback: "English"

    Args:
        explicit_language: User-specified language (highest priority)
        document_text: Document content for auto-detection
        env_default: Whether to check .env DEFAULT_LANGUAGE (default: True)

    Returns:
        Language name string (e.g., "Bangla", "English")

    Examples:
        >>> # Scenario 1: Explicit override
        >>> get_language_with_fallback(explicit_language="English", document_text="বাংলা")
        'English'

        >>> # Scenario 2: Auto-detection
        >>> get_language_with_fallback(document_text="এটি বাংলা")
        'Bangla'

        >>> # Scenario 3: .env fallback (if DEFAULT_LANGUAGE=Bangla)
        >>> get_language_with_fallback()
        'Bangla'

        >>> # Scenario 4: Hardcoded fallback
        >>> get_language_with_fallback(env_default=False)
        'English'
    """
    # Priority 1: Explicit parameter
    if explicit_language is not None:
        return explicit_language

    # Priority 2: Auto-detect from document
    if document_text:
        detected_lang = detect_document_language(document_text)
        if detected_lang:
            return detected_lang

    # Priority 3: Environment variable
    if env_default:
        from bigrag.config import config
        if config.default_language:
            return config.default_language

    # Priority 4: Hardcoded fallback
    return "English"
