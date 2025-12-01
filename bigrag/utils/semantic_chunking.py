"""
Semantic Chunking Utilities for BiG-RAG

Sophisticated semantic boundary-aware chunking with asymmetric overlap.
Ported from TableAwareChunker to provide robust chunking for modular system.
"""

from typing import List


class SemanticChunkingEngine:
    """
    Advanced semantic chunking with paragraph and sentence boundary awareness.

    Features:
    - Paragraph-level accumulation with tolerance factor (30% overflow allowed)
    - Asymmetric overlap (depends on chunk position)
    - Sentence-level splitting for oversized paragraphs
    - Bilingual sentence detection (English + Bangla)
    """

    @staticmethod
    def chunk_with_semantic_boundaries(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100,
        tolerance_factor: float = 1.3
    ) -> List[str]:
        """
        Chunk text respecting semantic boundaries.

        CRITICAL CLARIFICATIONS (from Technical Spec):

        1. ACCUMULATION DECISION: Decision made BEFORE adding next paragraph
           - If current + next <= chunk_size: Keep accumulating
           - If chunk_size < current + next <= chunk_size * tolerance:
             * If current >= chunk_size: Flush now (already large enough)
             * If current < chunk_size: Allow overflow (preserve paragraph)
           - If current + next > chunk_size * tolerance: MUST flush (hard limit)

        2. ASYMMETRIC OVERLAP: Depends on chunk position
           - First chunk: 0 before + overlap after = overlap total
           - Middle chunks: overlap before + overlap after = overlap * 2 total
           - Last chunk: overlap before + 0 after = overlap total
           - Single chunk: 0 before + 0 after = 0 total

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in tokens (default: 1000)
            overlap: Overlap in tokens on EACH side (default: 100)
            tolerance_factor: Overflow tolerance multiplier (default: 1.3 = 30%)

        Returns:
            List of text chunks with semantic boundaries preserved

        Example:
            Input: 3 paragraphs (800, 400, 900 tokens)
            Output: 2 chunks
                Chunk 1: Para1 (800) + Para2 (400) = 1200 tokens [within tolerance]
                Chunk 2: Para3 (900) + overlap from Para2 (100) = 1000 tokens
        """
        from bigrag.utils import count_tokens_fast, split_by_paragraphs

        if not text or not text.strip():
            return []

        # Step 1: Split by paragraphs (double newline)
        paragraphs = split_by_paragraphs(text)

        if not paragraphs:
            return []

        # Step 2: Accumulate paragraphs into chunks with semantic boundaries
        chunks = []
        current_chunk_paragraphs = []
        current_tokens = 0

        max_allowed = int(chunk_size * tolerance_factor)  # 1300 tokens

        for para_idx, para in enumerate(paragraphs):
            para_tokens = count_tokens_fast(para)

            # Case 1: Paragraph fits within tolerance as standalone
            if para_tokens <= max_allowed:
                # Check if adding this paragraph would exceed limits
                new_total = current_tokens + para_tokens

                # CRITICAL: Accumulation decision logic
                if new_total <= chunk_size:
                    # Under target - keep accumulating
                    current_chunk_paragraphs.append(para)
                    current_tokens = new_total

                elif new_total <= max_allowed:
                    # In overflow zone (1000-1300)
                    if current_tokens >= chunk_size:
                        # Current already large enough - flush it
                        if current_chunk_paragraphs:
                            chunks.append('\n\n'.join(current_chunk_paragraphs))
                        # Start new chunk with this paragraph
                        current_chunk_paragraphs = [para]
                        current_tokens = para_tokens
                    else:
                        # Current still small - allow overflow for coherence
                        current_chunk_paragraphs.append(para)
                        current_tokens = new_total

                else:
                    # Exceeds hard limit (> 1300) - MUST flush
                    if current_chunk_paragraphs:
                        chunks.append('\n\n'.join(current_chunk_paragraphs))
                    # Start new chunk with this paragraph
                    current_chunk_paragraphs = [para]
                    current_tokens = para_tokens

            else:
                # Case 2: Paragraph too large (> 1300 tokens) - split by sentences
                # First, flush current chunk if any
                if current_chunk_paragraphs:
                    chunks.append('\n\n'.join(current_chunk_paragraphs))
                    current_chunk_paragraphs = []
                    current_tokens = 0

                # Split large paragraph into sentence-based chunks
                sentence_chunks = SemanticChunkingEngine._split_paragraph_by_sentences(
                    para,
                    chunk_size=chunk_size,
                    max_allowed=max_allowed
                )
                chunks.extend(sentence_chunks)

        # Flush last chunk
        if current_chunk_paragraphs:
            chunks.append('\n\n'.join(current_chunk_paragraphs))

        # Step 3: Add asymmetric overlap
        chunks_with_overlap = SemanticChunkingEngine._add_asymmetric_overlap(chunks, overlap)

        return chunks_with_overlap

    @staticmethod
    def _split_paragraph_by_sentences(
        paragraph: str,
        chunk_size: int,
        max_allowed: int
    ) -> List[str]:
        """
        Split a large paragraph into chunks at sentence boundaries.

        Handles both English (. ! ?) and Bengali (।) sentence endings.

        Args:
            paragraph: Large paragraph to split
            chunk_size: Target chunk size in tokens
            max_allowed: Maximum allowed tokens (with tolerance)

        Returns:
            List of sentence-based chunks
        """
        from bigrag.utils import split_by_sentences, count_tokens_fast

        sentences = split_by_sentences(paragraph)

        if not sentences:
            # Fallback: return as-is if no sentence boundaries found
            return [paragraph]

        chunks = []
        current_chunk_sentences = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = count_tokens_fast(sent)

            if current_tokens + sent_tokens <= max_allowed and current_chunk_sentences:
                # Can add to current chunk
                current_chunk_sentences.append(sent)
                current_tokens += sent_tokens
            else:
                # Need to flush current chunk
                if current_chunk_sentences:
                    chunks.append(' '.join(current_chunk_sentences))

                # Start new chunk with this sentence
                current_chunk_sentences = [sent]
                current_tokens = sent_tokens

        # Flush last chunk
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))

        return chunks

    @staticmethod
    def _add_asymmetric_overlap(
        chunks: List[str],
        overlap: int
    ) -> List[str]:
        """
        Add asymmetric overlap to chunks.

        Asymmetry rules:
        - First chunk: 0 before + overlap after
        - Middle chunks: overlap before + overlap after
        - Last chunk: overlap before + 0 after
        - Single chunk: 0 before + 0 after

        Args:
            chunks: List of non-overlapping chunks
            overlap: Number of tokens to overlap on each side

        Returns:
            List of chunks with asymmetric overlap added
        """
        from bigrag.utils import get_overlap_text

        if not chunks:
            return []

        if len(chunks) == 1:
            # Single chunk - no overlap
            return chunks

        overlapped_chunks = []

        for idx, chunk in enumerate(chunks):
            is_first = (idx == 0)
            is_last = (idx == len(chunks) - 1)

            # Get overlap from previous chunk (if not first)
            before_overlap = ""
            if not is_first:
                before_overlap = get_overlap_text(chunks[idx - 1], overlap, direction='end')

            # Get overlap from next chunk (if not last)
            after_overlap = ""
            if not is_last:
                after_overlap = get_overlap_text(chunks[idx + 1], overlap, direction='start')

            # Build overlapped chunk
            overlapped_chunk = before_overlap + chunk + after_overlap
            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks
