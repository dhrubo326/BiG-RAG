#!/usr/bin/env python3
"""
Convert SingleTopic documents.csv to corpus.jsonl format for BiG-RAG indexing.

This script reads the documents from datasets/SingleTopic/raw/documents.csv
and converts them to the corpus.jsonl format expected by script_build.py.

Usage:
    python scripts/convert_singletopic_docs.py

Input:
    datasets/SingleTopic/raw/documents.csv
    Schema: index, source_url, text

Output:
    datasets/SingleTopic/raw/corpus.jsonl
    Schema: {"id": str, "contents": str, "metadata": {...}}
"""

import pandas as pd
import json
import os
from pathlib import Path


def convert_documents_to_corpus(
    input_csv: str = "datasets/SingleTopic/raw/documents.csv",
    output_jsonl: str = "datasets/SingleTopic/raw/corpus.jsonl"
):
    """
    Convert documents.csv to corpus.jsonl format.

    Args:
        input_csv: Path to input CSV file
        output_jsonl: Path to output JSONL file
    """
    # Ensure paths exist
    input_path = Path(input_csv)
    output_path = Path(output_jsonl)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading documents from: {input_csv}")

    # Read CSV
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    # Validate required columns
    required_cols = ['index', 'source_url', 'text']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Found: {df.columns.tolist()}")

    print(f"Found {len(df)} documents")

    # Convert to corpus.jsonl
    documents_written = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            # Create document in corpus format
            doc = {
                "id": str(row['index']),
                "contents": row['text'],
                "metadata": {
                    "source_url": row['source_url'],
                    "title": f"Document {row['index']}",
                    "source": "SingleTopic",
                    "original_index": int(row['index'])
                }
            }

            # Write as JSON line
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            documents_written += 1

    print(f"✅ Successfully converted {documents_written} documents")
    print(f"Output saved to: {output_jsonl}")
    print(f"\nNext steps:")
    print(f"  1. Build knowledge graph: python script_build.py --data_source SingleTopic")
    print(f"  2. Start API server: python script_api.py --data_source SingleTopic")

    return documents_written


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SingleTopic documents.csv to corpus.jsonl"
    )
    parser.add_argument(
        '--input',
        default='datasets/SingleTopic/raw/documents.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        default='datasets/SingleTopic/raw/corpus.jsonl',
        help='Output JSONL file path'
    )

    args = parser.parse_args()

    try:
        convert_documents_to_corpus(args.input, args.output)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
