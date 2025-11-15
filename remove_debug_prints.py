"""Remove all debug print statements from code"""
import re

# Remove debug prints from operate.py
with open('bigrag/operate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove lines that start with whitespace + print(f"[...
content = re.sub(r'^\s+print\(f"\[.*?\)\n', '', content, flags=re.MULTILINE)

with open('bigrag/operate.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Removed debug prints from operate.py")

# Also check utils.py
try:
    with open('bigrag/utils.py', 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(r'^\s+print\(f"\[.*?\)\n', '', content, flags=re.MULTILINE)

    with open('bigrag/utils.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Removed debug prints from utils.py")
except Exception as e:
    print(f"Error with utils.py: {e}")

print("Done!")
