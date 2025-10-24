"""
Convert Custom Text Files to BiG-RAG Corpus Format

This script converts one or multiple text files into the corpus.jsonl format
required by BiG-RAG for knowledge graph construction.

Usage:
    # Single file
    python convert_text_to_corpus.py --input myfile.txt --output datasets/my_data/raw/corpus.jsonl

    # Multiple files
    python convert_text_to_corpus.py --input file1.txt file2.txt file3.txt --output datasets/my_data/raw/corpus.jsonl

    # All text files in directory
    python convert_text_to_corpus.py --input-dir documents/ --output datasets/my_data/raw/corpus.jsonl

    # With custom splitting (large documents)
    python convert_text_to_corpus.py --input large.txt --output corpus.jsonl --split-by-paragraphs
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import List
import sys

def compute_doc_id(content: str, prefix: str = "doc") -> str:
    """Generate unique ID from content hash"""
    hash_obj = hashlib.md5(content.encode('utf-8'))
    return f"{prefix}-{hash_obj.hexdigest()[:16]}"


def split_by_paragraphs(text: str, min_length: int = 100) -> List[str]:
    """
    Split text by paragraphs (double newlines)

    Args:
        text: Input text
        min_length: Minimum character length for a chunk (shorter ones are merged)

    Returns:
        List of text chunks
    """
    # Split by double newlines (paragraphs)
    raw_chunks = text.split('\n\n')

    # Filter and merge small chunks
    chunks = []
    buffer = ""

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        buffer += chunk + "\n\n"

        if len(buffer) >= min_length:
            chunks.append(buffer.strip())
            buffer = ""

    # Add remaining buffer
    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


def split_by_sentences(text: str, max_sentences: int = 10) -> List[str]:
    """
    Split text into chunks of N sentences

    Args:
        text: Input text
        max_sentences: Maximum sentences per chunk

    Returns:
        List of text chunks
    """
    import re

    # Simple sentence splitter (splits on . ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)

        if len(current_chunk) >= max_sentences:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    # Add remaining sentences
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def convert_file(input_path: Path, split_mode: str = "whole", **split_kwargs) -> List[dict]:
    """
    Convert a single text file to corpus format

    Args:
        input_path: Path to input text file
        split_mode: "whole", "paragraphs", or "sentences"
        **split_kwargs: Additional arguments for splitting

    Returns:
        List of document dictionaries
    """
    print(f"Converting: {input_path}")

    # Read file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except UnicodeDecodeError:
        # Try with different encodings
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(input_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                print(f"  (Read with {encoding} encoding)")
                break
            except:
                continue
        else:
            print(f"  ERROR: Could not read file with any encoding")
            return []

    if not content:
        print(f"  WARNING: File is empty")
        return []

    # Split content based on mode
    if split_mode == "whole":
        chunks = [content]
    elif split_mode == "paragraphs":
        min_length = split_kwargs.get('min_paragraph_length', 100)
        chunks = split_by_paragraphs(content, min_length=min_length)
    elif split_mode == "sentences":
        max_sentences = split_kwargs.get('max_sentences_per_chunk', 10)
        chunks = split_by_sentences(content, max_sentences=max_sentences)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    # Create documents
    documents = []
    filename = input_path.stem  # Filename without extension

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        # Generate unique ID
        doc_id = compute_doc_id(chunk)

        # Create document
        if len(chunks) == 1:
            title = filename
        else:
            title = f"{filename} (Part {i+1}/{len(chunks)})"

        doc = {
            "id": doc_id,
            "contents": chunk.strip(),
            "title": title,
            "source_file": str(input_path),
            "chunk_index": i if len(chunks) > 1 else None
        }

        documents.append(doc)

    print(f"  Created {len(documents)} document(s)")
    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Convert text files to BiG-RAG corpus format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file (keep whole)
  python convert_text_to_corpus.py --input document.txt --output datasets/my_data/raw/corpus.jsonl

  # Multiple files
  python convert_text_to_corpus.py --input file1.txt file2.txt --output corpus.jsonl

  # Directory of files
  python convert_text_to_corpus.py --input-dir documents/ --output corpus.jsonl

  # Split large file by paragraphs
  python convert_text_to_corpus.py --input large.txt --split-by-paragraphs --output corpus.jsonl

  # Split by sentences (10 sentences per chunk)
  python convert_text_to_corpus.py --input book.txt --split-by-sentences --max-sentences 15 --output corpus.jsonl
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', nargs='+', help='Input text file(s)')
    input_group.add_argument('--input-dir', help='Directory containing text files')

    # Output
    parser.add_argument('--output', required=True, help='Output corpus.jsonl file')

    # Splitting options
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument('--split-by-paragraphs', action='store_true',
                            help='Split documents by paragraphs (double newlines)')
    split_group.add_argument('--split-by-sentences', action='store_true',
                            help='Split documents by sentences')

    # Splitting parameters
    parser.add_argument('--min-paragraph-length', type=int, default=100,
                       help='Minimum paragraph length in characters (default: 100)')
    parser.add_argument('--max-sentences', type=int, default=10,
                       help='Maximum sentences per chunk (default: 10)')

    # Other options
    parser.add_argument('--extensions', nargs='+', default=['.txt', '.md', '.text'],
                       help='File extensions to process (default: .txt .md .text)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite output file if it exists')

    args = parser.parse_args()

    # Determine split mode
    if args.split_by_paragraphs:
        split_mode = "paragraphs"
    elif args.split_by_sentences:
        split_mode = "sentences"
    else:
        split_mode = "whole"

    print("="*80)
    print("BiG-RAG Text to Corpus Converter")
    print("="*80)
    print()

    # Check output file
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"ERROR: Output file already exists: {output_path}")
        print("Use --overwrite to replace it")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect input files
    input_files = []

    if args.input:
        # Direct file list
        for filepath in args.input:
            path = Path(filepath)
            if not path.exists():
                print(f"WARNING: File not found: {path}")
                continue
            if not path.is_file():
                print(f"WARNING: Not a file: {path}")
                continue
            input_files.append(path)

    elif args.input_dir:
        # Directory scan
        dir_path = Path(args.input_dir)
        if not dir_path.exists():
            print(f"ERROR: Directory not found: {dir_path}")
            sys.exit(1)

        print(f"Scanning directory: {dir_path}")
        print(f"Looking for extensions: {', '.join(args.extensions)}")
        print()

        for ext in args.extensions:
            input_files.extend(dir_path.glob(f"*{ext}"))

        if not input_files:
            print(f"ERROR: No files found with extensions {args.extensions}")
            sys.exit(1)

    print(f"Found {len(input_files)} file(s) to process")
    print(f"Split mode: {split_mode}")
    print()

    # Convert all files
    all_documents = []

    for input_path in sorted(input_files):
        docs = convert_file(
            input_path,
            split_mode=split_mode,
            min_paragraph_length=args.min_paragraph_length,
            max_sentences_per_chunk=args.max_sentences
        )
        all_documents.extend(docs)

    print()
    print(f"Total documents created: {len(all_documents)}")
    print()

    # Write to JSONL
    print(f"Writing to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in all_documents:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')

    print(f"✓ Successfully created corpus with {len(all_documents)} documents")
    print()
    print("Next steps:")
    print(f"  1. Review the corpus: {output_path}")
    print(f"  2. Build knowledge graph:")
    print(f"     python build_kg_from_corpus.py --data-source {output_path.parent.parent.name}")
    print()


if __name__ == "__main__":
    main()
