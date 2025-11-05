"""
Utility Functions

Markdown processing, validation, and helper functions
"""

import re
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def process_markdown(md_content: str) -> str:
    """
    Convert Markdown to plain text for indexing

    Preserves structure while removing syntax

    Args:
        md_content: Markdown text

    Returns:
        Plain text suitable for knowledge graph construction
    """
    try:
        import markdown
        from bs4 import BeautifulSoup

        # Convert MD → HTML
        html = markdown.markdown(
            md_content,
            extensions=[
                'extra',           # Tables, fenced code, etc.
                'nl2br',           # Newline to <br>
                'sane_lists'       # Better list handling
            ]
        )

        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Extract text
        text = soup.get_text(separator='\n')

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = text.strip()

        return text

    except Exception as e:
        logger.error(f"Markdown processing failed: {e}. Returning raw content.")
        # Fallback: use raw content
        return md_content


def validate_file_upload(
    file_bytes: bytes,
    filename: str,
    max_size_mb: int = 50
) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file

    Args:
        file_bytes: File content bytes
        filename: Original filename
        max_size_mb: Maximum file size in MB

    Returns:
        (is_valid, error_message)
    """
    # Check size
    size_mb = len(file_bytes) / 1024 / 1024

    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f} MB (max {max_size_mb} MB)"

    # Check extension
    if not filename.endswith(('.txt', '.md')):
        return False, "Only .txt and .md files are supported"

    # Check UTF-8 encoding
    try:
        file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return False, "File must be UTF-8 encoded"

    # Check not empty
    content = file_bytes.decode('utf-8').strip()
    if len(content) == 0:
        return False, "File is empty"

    return True, None


def sanitize_metadata(metadata: dict) -> dict:
    """
    Remove potentially dangerous fields from metadata

    Args:
        metadata: User-provided metadata

    Returns:
        Sanitized metadata
    """
    dangerous_keys = ['__proto__', 'constructor', 'prototype', '__init__', '__class__']

    return {
        k: v for k, v in metadata.items()
        if k not in dangerous_keys
    }


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "12.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to maximum length with ellipsis

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length].rstrip() + "..."
