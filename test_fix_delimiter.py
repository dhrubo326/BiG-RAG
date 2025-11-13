"""Test fix_delimiter_corruption function"""
from bigrag.utils import fix_delimiter_corruption

# Test with actual LLM output format
test_input = '"relation"<<|>>"Atletico Madrid, under Diego Simeone, remains competitive in La Liga."<<|>>8'

print(f"Input:  {repr(test_input)}")

result = fix_delimiter_corruption(test_input, "<|>")

print(f"Output: {repr(result)}")
print(f"\nDid it work? {test_input != result}")

# Check what pattern we're looking for
core = "|"
pattern = f"<<{core}>>"
print(f"\nPattern to match: {repr(pattern)}")
print(f"Pattern in input: {pattern in test_input}")

# Manual replacement
manual_fix = test_input.replace("<<|>>", "<|>")
print(f"\nManual fix: {repr(manual_fix)}")
