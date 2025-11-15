"""
Debug script to test entity extraction with current prompts
"""
import asyncio
from bigrag import BiGRAG
from bigrag.prompt import PROMPTS

async def test_extraction():
    # Initialize BiG-RAG (not actually needed for this test)
    # rag = BiGRAG(
    #     working_dir="./expr/football_debug",
    #     enable_llm_cache=False
    # )

    # Test text (short football excerpt)
    test_text = """
[DOCUMENT CONTEXT]
Title: Football World Update: 2024 Season Highlights
Metadata: category: sports, tags: football, messi, premier-league

[CHUNK CONTENT]
Lionel Messi, widely regarded as one of the greatest football players of all time, continues to make headlines in Major League Soccer with Inter Miami. The Argentine superstar joined the club in July 2023 and has transformed the team's fortunes dramatically. Messi scored 11 goals in his first 14 matches.
"""

    # Get the extraction prompt
    entity_extract_prompt = PROMPTS["entity_extraction"]

    # Format the prompt
    from bigrag.constants import DEFAULT_ENTITY_TYPES

    context = {
        "entity_types": ", ".join(DEFAULT_ENTITY_TYPES),
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>",
        "language": "English",
        "examples": "",  # Skip examples for brevity
        "input_text": test_text
    }

    formatted_prompt = entity_extract_prompt.format(**context)

    # Save prompt to file (avoid encoding issues)
    with open("debug_prompt.txt", "w", encoding="utf-8") as f:
        f.write("FORMATTED PROMPT:\n")
        f.write("="*80 + "\n")
        f.write(formatted_prompt[:1500] + "...\n")

    # Call LLM
    print("="*80)
    print("CALLING LLM...")
    print("="*80)

    from bigrag.llm import gpt_4o_mini_complete

    try:
        response = await gpt_4o_mini_complete(formatted_prompt)
        print("\n[OK] LLM call successful")

        # Save response to file
        with open("debug_response.txt", "w", encoding="utf-8") as f:
            f.write("LLM RESPONSE:\n")
            f.write("="*80 + "\n")
            f.write(response)
            f.write("\n" + "="*80 + "\n")

        # Try parsing
        from bigrag.utils import split_string_by_multi_markers
        import re

        records = split_string_by_multi_markers(
            response,
            [context["record_delimiter"], context["completion_delimiter"]]
        )

        print(f"\nParsed {len(records)} records")

        for i, record in enumerate(records[:5], 1):
            record_match = re.search(r"\((.*)\)", record)
            if record_match:
                record_content = record_match.group(1)
                parts = split_string_by_multi_markers(record_content, [context["tuple_delimiter"]])
                print(f"\nRecord {i}: {len(parts)} parts")
                print(f"  Type: {parts[0] if parts else 'N/A'}")
                if len(parts) > 1:
                    print(f"  Content: {parts[1][:60]}...")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_extraction())
