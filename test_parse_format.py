"""Test actual LLM output format"""
from bigrag.utils import split_string_by_multi_markers
import re

# Actual LLM output from build
sample = '("relation"<|>"Atletico Madrid remains competitive in La Liga under Diego Simeone."<|>8)##'

# Remove ##
sample = sample.replace('##', '')

# Extract from parentheses
record_match = re.search(r"\((.*)\)", sample)
if record_match:
    record_content = record_match.group(1)
    print(f"Record content: {repr(record_content)}")

    # Split by delimiter
    parts = split_string_by_multi_markers(record_content, ["<|>"])
    print(f"\nNumber of parts: {len(parts)}")
    for i, part in enumerate(parts):
        print(f"Part {i}: {repr(part)}")

    # Check first field
    print(f"\nFirst field == '\"relation\"': {parts[0] == '\"relation\"'}")
    print(f"First field value: {repr(parts[0])}")
    print(f"Expected value: {repr('\"relation\"')}")
