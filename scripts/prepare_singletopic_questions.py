#!/usr/bin/env python3
"""
Unify SingleTopic question CSV files into single evaluation-ready format.

This script takes the three separate question CSV files from the SingleTopic dataset
and combines them into a single unified CSV file with standardized schema.

Input Files:
    - multi_passage_answer_questions.csv: Questions requiring multiple documents
    - single_passage_answer_questions.csv: Questions requiring single document
    - no_answer_questions.csv: Questions with NO answer in corpus

Output File:
    - all_questions_unified.csv: Combined questions in unified schema

Unified Schema:
    question, golden_answer, document_index, question_type

Usage:
    python scripts/prepare_singletopic_questions.py
"""

import pandas as pd
import os
from pathlib import Path


def unify_question_csvs(
    input_dir: str = "datasets/SingleTopic/raw",
    output_dir: str = "datasets/SingleTopic/processed",
    output_filename: str = "all_questions_unified.csv"
):
    """
    Unify three question CSV files into single evaluation-ready format.

    Args:
        input_dir: Directory containing input CSV files
        output_dir: Directory for output unified CSV
        output_filename: Name of output file

    Returns:
        Number of total questions unified
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Define input files
    multi_passage_file = input_path / "multi_passage_answer_questions.csv"
    single_passage_file = input_path / "single_passage_answer_questions.csv"
    no_answer_file = input_path / "no_answer_questions.csv"

    # Check all files exist
    missing_files = []
    for file_path, name in [
        (multi_passage_file, "multi_passage_answer_questions.csv"),
        (single_passage_file, "single_passage_answer_questions.csv"),
        (no_answer_file, "no_answer_questions.csv")
    ]:
        if not file_path.exists():
            missing_files.append(name)

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {input_dir}: {', '.join(missing_files)}"
        )

    print(f"Reading question files from: {input_dir}")

    # Load CSV files
    try:
        multi_df = pd.read_csv(multi_passage_file)
        single_df = pd.read_csv(single_passage_file)
        no_answer_df = pd.read_csv(no_answer_file)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV files: {e}")

    print(f"  - Multi-passage questions: {len(multi_df)}")
    print(f"  - Single-passage questions: {len(single_df)}")
    print(f"  - No-answer questions: {len(no_answer_df)}")

    # Add question_type column
    multi_df = multi_df.copy()
    single_df = single_df.copy()
    no_answer_df = no_answer_df.copy()

    multi_df['question_type'] = 'multi_passage'
    single_df['question_type'] = 'single_passage'
    no_answer_df['question_type'] = 'no_answer'

    # Add empty 'answer' column to no_answer questions
    no_answer_df['answer'] = ''

    # Rename 'answer' column to 'golden_answer' for all
    multi_df = multi_df.rename(columns={'answer': 'golden_answer'})
    single_df = single_df.rename(columns={'answer': 'golden_answer'})
    no_answer_df = no_answer_df.rename(columns={'answer': 'golden_answer'})

    # Combine all dataframes
    unified_df = pd.concat([multi_df, single_df, no_answer_df], ignore_index=True)

    # Reorder columns to standard format
    unified_df = unified_df[['question', 'golden_answer', 'document_index', 'question_type']]

    # Save to output file
    output_file = output_path / output_filename
    unified_df.to_csv(output_file, index=False, encoding='utf-8')

    total_questions = len(unified_df)

    print(f"\n[OK] Successfully unified {total_questions} questions")
    print(f"Output saved to: {output_file}")
    print(f"\nQuestion type breakdown:")
    print(f"  - Multi-passage: {len(multi_df)} ({len(multi_df)/total_questions*100:.1f}%)")
    print(f"  - Single-passage: {len(single_df)} ({len(single_df)/total_questions*100:.1f}%)")
    print(f"  - No-answer: {len(no_answer_df)} ({len(no_answer_df)/total_questions*100:.1f}%)")
    print(f"\nNext steps:")
    print(f"  1. Process questions: POST /eval/batch_generate")
    print(f"  2. Evaluate results: POST /eval/evaluate_results")

    return total_questions


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unify SingleTopic question CSV files into evaluation-ready format"
    )
    parser.add_argument(
        '--input-dir',
        default='datasets/SingleTopic/raw',
        help='Directory containing input CSV files'
    )
    parser.add_argument(
        '--output-dir',
        default='datasets/SingleTopic/processed',
        help='Directory for output unified CSV'
    )
    parser.add_argument(
        '--output-file',
        default='all_questions_unified.csv',
        help='Name of output file'
    )

    args = parser.parse_args()

    try:
        unify_question_csvs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            output_filename=args.output_file
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
