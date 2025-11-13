"""Test parsing a single record"""
from bigrag.utils import split_string_by_multi_markers
import re

# Sample record from LLM
record_text = '("relation"<|>Lionel Messi continues to make headlines in Major League Soccer with Inter Miami.<|>10)##'

# Remove the ## delimiter
record_text = record_text.replace('##', '')

# Extract content from parentheses
record_match = re.search(r"\((.*)\)", record_text)
if record_match:
    record_content = record_match.group(1)
    print(f"Record content: {record_content[:80]}...")
    
    # Split by delimiter
    parts = split_string_by_multi_markers(record_content, ["<|>"])
    print(f"\nNumber of parts: {len(parts)}")
    for i, part in enumerate(parts):
        print(f"Part {i}: {repr(part[:50])}")
else:
    print("No match!")
