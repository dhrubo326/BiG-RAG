"""
Validate SingleTopic Dataset Structure

Checks:
1. corpus.jsonl format (id, contents, metadata)
2. Questions CSV format (question, golden_answer, document_index, question_type)
3. Document index consistency (questions reference valid documents)
4. Metadata completeness
"""

import json
import csv
import sys
from pathlib import Path

def validate_corpus(corpus_path: Path):
    """Validate corpus.jsonl structure"""
    print("\n" + "="*80)
    print("VALIDATING CORPUS.JSONL")
    print("="*80)

    if not corpus_path.exists():
        print(f"[ERROR] Corpus file not found: {corpus_path}")
        return False

    errors = []
    warnings = []
    documents = {}

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line)

                # Check required fields
                if 'id' not in doc:
                    errors.append(f"Line {line_num}: Missing 'id' field")
                    continue

                if 'contents' not in doc:
                    errors.append(f"Line {line_num}: Missing 'contents' field")
                    continue

                doc_id = doc['id']

                # Check for duplicates
                if doc_id in documents:
                    errors.append(f"Line {line_num}: Duplicate document ID '{doc_id}'")

                documents[doc_id] = {
                    'line': line_num,
                    'content_length': len(doc['contents']),
                    'has_metadata': 'metadata' in doc,
                    'has_title': 'title' in doc.get('metadata', {}) or 'title' in doc
                }

                # Check metadata
                if 'metadata' not in doc:
                    warnings.append(f"Line {line_num} (ID: {doc_id}): No metadata field")
                else:
                    metadata = doc['metadata']
                    if 'title' not in metadata and 'title' not in doc:
                        warnings.append(f"Line {line_num} (ID: {doc_id}): No title in metadata")
                    if 'source_url' not in metadata:
                        warnings.append(f"Line {line_num} (ID: {doc_id}): No source_url in metadata")

                # Check content length
                if len(doc['contents']) < 100:
                    warnings.append(f"Line {line_num} (ID: {doc_id}): Very short content ({len(doc['contents'])} chars)")

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
            except Exception as e:
                errors.append(f"Line {line_num}: Unexpected error - {e}")

    # Print results
    print(f"\n[CORPUS VALIDATION]")
    print(f"  Total documents: {len(documents)}")
    print(f"  Document IDs: {sorted(documents.keys(), key=lambda x: int(x) if x.isdigit() else x)}")

    if errors:
        print(f"\n[ERRORS] Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print(f"\n[ERRORS] None found")

    if warnings:
        print(f"\n[WARNINGS] Found {len(warnings)} warnings:")
        for warning in warnings[:10]:  # Show first 10
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    else:
        print(f"\n[WARNINGS] None found")

    return len(errors) == 0, documents


def validate_questions(questions_path: Path, valid_doc_ids: set):
    """Validate questions CSV structure"""
    print("\n" + "="*80)
    print("VALIDATING QUESTIONS CSV")
    print("="*80)

    if not questions_path.exists():
        print(f"[ERROR] Questions file not found: {questions_path}")
        return False

    errors = []
    warnings = []
    questions = []

    with open(questions_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)

        # Check required columns
        required_columns = ['question', 'golden_answer', 'document_index', 'question_type']
        if not all(col in reader.fieldnames for col in required_columns):
            missing = [col for col in required_columns if col not in reader.fieldnames]
            errors.append(f"Missing required columns: {missing}")
            print(f"\n[ERROR] Missing columns: {missing}")
            print(f"Found columns: {reader.fieldnames}")
            return False

        for row_num, row in enumerate(reader, 2):  # Start at 2 (1 is header)
            try:
                question = row.get('question', '').strip()
                golden_answer = row.get('golden_answer', '').strip()
                document_index = row.get('document_index', '').strip()
                question_type = row.get('question_type', '').strip()

                # Check for empty fields
                if not question:
                    errors.append(f"Row {row_num}: Empty question")
                # Empty golden_answer is OK for no_answer questions
                if not golden_answer and question_type != 'no_answer':
                    errors.append(f"Row {row_num}: Empty golden_answer (not a no_answer question)")
                if not document_index:
                    errors.append(f"Row {row_num}: Empty document_index")
                if not question_type:
                    warnings.append(f"Row {row_num}: Empty question_type")

                # Check document_index validity
                if document_index and document_index not in valid_doc_ids:
                    errors.append(f"Row {row_num}: Invalid document_index '{document_index}' (not in corpus)")

                # Check question types
                valid_types = ['single_passage', 'multi_passage', 'no_answer']
                if question_type and question_type not in valid_types:
                    warnings.append(f"Row {row_num}: Unexpected question_type '{question_type}' (expected: {valid_types})")

                questions.append({
                    'row': row_num,
                    'question': question,
                    'doc_index': document_index,
                    'type': question_type
                })

            except Exception as e:
                errors.append(f"Row {row_num}: Unexpected error - {e}")

    # Statistics
    type_counts = {}
    for q in questions:
        qtype = q['type']
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    doc_counts = {}
    for q in questions:
        doc = q['doc_index']
        doc_counts[doc] = doc_counts.get(doc, 0) + 1

    # Print results
    print(f"\n[QUESTIONS VALIDATION]")
    print(f"  Total questions: {len(questions)}")
    print(f"  Question types: {type_counts}")
    print(f"  Questions per document: {dict(sorted(doc_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]))}")

    if errors:
        print(f"\n[ERRORS] Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print(f"\n[ERRORS] None found")

    if warnings:
        print(f"\n[WARNINGS] Found {len(warnings)} warnings:")
        for warning in warnings[:10]:
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    else:
        print(f"\n[WARNINGS] None found")

    return len(errors) == 0


def main():
    """Main validation function"""
    print("="*80)
    print("SINGLETOPIC DATASET VALIDATOR")
    print("="*80)

    # Paths
    base_dir = Path(__file__).parent
    corpus_path = base_dir / "datasets" / "SingleTopic" / "raw" / "corpus.jsonl"
    questions_path = base_dir / "datasets" / "SingleTopic" / "processed" / "all_questions_unified.csv"

    # Validate corpus
    corpus_valid, documents = validate_corpus(corpus_path)

    # Validate questions
    questions_valid = validate_questions(questions_path, set(documents.keys()))

    # Final verdict
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    if corpus_valid and questions_valid:
        print("\n[OK] All validations passed!")
        print("\n[NEXT STEPS]")
        print("  1. Build knowledge graph:")
        print("     python script_build.py --data_source SingleTopic --batch_size 5")
        print("")
        print("  2. Start API server:")
        print("     python script_api.py --data_source SingleTopic")
        print("")
        print("  3. Test retrieval:")
        print("     curl -X POST http://localhost:8001/search \\")
        print("       -H \"Content-Type: application/json\" \\")
        print("       -d '{\"queries\": [\"test query\"]}'")
        print("")
        return 0
    else:
        print("\n[FAIL] Validation failed!")
        print("  Please fix the errors above before proceeding.")
        print("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
