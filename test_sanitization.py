"""Test sanitization with actual LLM format"""
from bigrag.utils import sanitize_extracted_text

# Actual values from LLM output
relation_content_raw = '"Atletico Madrid remains competitive in La Liga under Diego Simeone."'
entity_name_raw = '"Atletico Madrid"'
entity_type_raw = '"organization"'
entity_desc_raw = '"Atletico Madrid is a professional football club located in Madrid, Spain, known for its competitive presence in La Liga."'

print("Testing sanitization:")
print(f"\n1. Relation content:")
print(f"   Input: {repr(relation_content_raw)}")
result = sanitize_extracted_text(relation_content_raw, "relation")
print(f"   Output: {repr(result)}")
print(f"   Is empty: {not result}")

print(f"\n2. Entity name:")
print(f"   Input: {repr(entity_name_raw)}")
result = sanitize_extracted_text(entity_name_raw, "entity_name")
print(f"   Output: {repr(result)}")
print(f"   Is empty: {not result}")

print(f"\n3. Entity type:")
print(f"   Input: {repr(entity_type_raw)}")
result = sanitize_extracted_text(entity_type_raw, "entity_type")
print(f"   Output: {repr(result)}")
print(f"   Is empty: {not result}")

print(f"\n4. Entity description:")
print(f"   Input: {repr(entity_desc_raw[:50])}...")
result = sanitize_extracted_text(entity_desc_raw, "description")
print(f"   Output: {repr(result[:50] if result else 'EMPTY')}...")
print(f"   Is empty: {not result}")
