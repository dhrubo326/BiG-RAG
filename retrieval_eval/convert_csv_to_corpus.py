"""
Convert CSV Documents to BiG-RAG Corpus Format

Specifically designed for the Single-Topic evaluation dataset format:
- Reads documents.csv (index, source_url, text columns)
- Converts to corpus.jsonl format required by BiG-RAG

Usage:
    python convert_csv_to_corpus.py --csv datasets/Single-Topic/raw/documents.csv
    python convert_csv_to_corpus.py --csv datasets/Single-Topic/raw/documents.csv --output datasets/Single-Topic/raw/corpus.jsonl
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys


def convert_csv_to_corpus(csv_path: Path, output_path: Path, overwrite: bool = False):
    """
    Convert documents.csv to corpus.jsonl format

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output corpus.jsonl file
        overwrite: Whether to overwrite existing output file
    """
    print("="*80)
    print("CSV to Corpus Converter for BiG-RAG")
    print("="*80)
    print()

    # Check input file
    if not csv_path.exists():
        print(f"ERROR: Input file not found: {csv_path}")
        sys.exit(1)

    # Check output file
    if output_path.exists() and not overwrite:
        print(f"ERROR: Output file already exists: {output_path}")
        print("Use --overwrite to replace it")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read CSV
    print(f"Reading CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        sys.exit(1)

    # Validate columns
    required_columns = ['index', 'text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Found {len(df)} documents")
    print()

    # Convert to corpus format
    documents = []
    for idx, row in df.iterrows():
        doc = {
            "id": str(row["index"]),
            "contents": str(row["text"]),
            "title": row.get("source_url", f"Document {row['index']}"),
            "metadata": {
                "source_url": row.get("source_url", ""),
                "original_index": int(row["index"])
            }
        }
        documents.append(doc)

    # Write to JSONL
    print(f"Writing corpus.jsonl: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')

    print(f"✓ Successfully created corpus with {len(documents)} documents")
    print()
    print("="*80)
    print("Next Steps:")
    print("="*80)
    print()
    print("1. Build Knowledge Graph:")
    print(f"   python script_build.py --data_source Single-Topic")
    print()
    print("2. Run Evaluation:")
    print(f"   python script_evaluate_single_topic.py")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert documents.csv to BiG-RAG corpus.jsonl format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python convert_csv_to_corpus.py --csv datasets/Single-Topic/raw/documents.csv
  python convert_csv_to_corpus.py --csv datasets/Single-Topic/raw/documents.csv --overwrite
        """
    )

    parser.add_argument('--csv', required=True, help='Input CSV file (documents.csv)')
    parser.add_argument('--output', help='Output corpus.jsonl file (default: same dir as CSV)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file if exists')

    args = parser.parse_args()

    # Determine paths
    csv_path = Path(args.csv)

    if args.output:
        output_path = Path(args.output)
    else:
        # Default: same directory as CSV, named corpus.jsonl
        output_path = csv_path.parent / "corpus.jsonl"

    # Convert
    convert_csv_to_corpus(csv_path, output_path, args.overwrite)


if __name__ == "__main__":
    main()
