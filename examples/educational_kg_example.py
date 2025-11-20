"""
Example: Building Educational Knowledge Graph with BiG-RAG

This example demonstrates how to use the educational pipeline to build
a knowledge graph from KUET/BUET admission documents.

Requirements:
- OpenAI API key (set in openai_api_key.txt or pass as argument)
- Documents in markdown format
"""

import asyncio
import os
from bigrag.educational_pipeline import build_educational_kg


async def main():
    # Example 1: Sample educational document (KUET CSE department)
    sample_document = """
# KUET Admission Information 2024-25

## Department Information

| বিভাগ/বিষয় | কোড | আসন সংখ্যা |
|------------|-----|-----------|
| কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং | CSE | ১২০ |
| ইলেক্ট্রিক্যাল এন্ড ইলেক্ট্রনিক ইঞ্জিনিয়ারিং | EEE | ১২০ |
| সিভিল ইঞ্জিনিয়ারিং | CE | ১২০ |
| মেকানিক্যাল ইঞ্জিনিয়ারিং | ME | ১২০ |

## Admission Requirements

- **SSC GPA**: ৪.০০ minimum
- **HSC GPA**: ৪.০০ minimum
- **Combined GPA**: ৮.০০ minimum

## Fees

| Category | Fee Amount |
|----------|-----------|
| Engineering | ১১০০ টাকা |
| Science | ১১০০ টাকা |

## Important Dates

- Application Start: ০১ ডিসেম্বর, ২০২৪
- Application Deadline: ১৫ ডিসেম্বর, ২০২৪
- Admission Test: ২৫ ডিসেম্বর, ২০২৪
"""

    # Read API key
    api_key_file = "openai_api_key.txt"
    if os.path.exists(api_key_file):
        with open(api_key_file) as f:
            api_key = f.read().strip()
    else:
        print(f"[ERROR] {api_key_file} not found. Please create it with your OpenAI API key.")
        return

    # Prepare documents and metadata
    documents = [sample_document]
    metadata = [
        {
            'title': 'KUET Admission 2024-25',
            'category': 'university',
            'tags': ['engineering', 'admission', 'KUET']
        }
    ]

    print("="*80)
    print("Educational Knowledge Graph Builder - Example")
    print("="*80)
    print(f"\nProcessing {len(documents)} document(s)...")
    print(f"API Key: {api_key[:10]}..." if api_key else "No API key")

    # Build knowledge graph
    rag, results = await build_educational_kg(
        markdown_documents=documents,
        document_metadata=metadata,
        api_key=api_key,
        working_dir="./expr/educational_kg_example",
        validation_level="STRICT"  # 99%+ accuracy
    )

    print("\n" + "="*80)
    print("Knowledge Graph Built Successfully!")
    print("="*80)

    # Show validation results
    for i, result in enumerate(results):
        print(f"\nDocument {i+1}: {metadata[i]['title']}")
        print(f"  Status: {result['validation']['overall_status']}")
        print(f"  Entities: {result['statistics']['total_entities']}")
        print(f"  Relations: {result['statistics']['total_relations']}")
        print(f"  Numeric Coverage: {result['statistics']['numeric_coverage']:.2%}")
        print(f"  Consistency: {result['statistics']['consistency_score']:.2%}")

    # Example queries
    print("\n" + "="*80)
    print("Example Queries")
    print("="*80)

    queries = [
        "কুয়েটে CSE বিভাগে কতটি আসন আছে?",
        "What is the admission fee for engineering?",
        "কখন ভর্তি পরীক্ষা অনুষ্ঠিত হবে?",
        "What is the minimum SSC GPA requirement?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        contexts = rag.query(query)
        print(f"Contexts retrieved: {len(contexts)}")
        if contexts:
            print(f"Top context: {contexts[0][:200]}...")

    print("\n" + "="*80)
    print("Example Complete!")
    print("="*80)
    print(f"\nKnowledge graph saved to: ./expr/educational_kg_example")
    print("You can now use this graph for question answering, chatbots, etc.")


if __name__ == "__main__":
    asyncio.run(main())
