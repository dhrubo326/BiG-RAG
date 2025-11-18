import asyncio
from bigrag.operate import preprocess_query
from bigrag.llm import gpt_4o_mini_complete

async def test_english():
    n, s = await preprocess_query("who is messi", "English", gpt_4o_mini_complete, {})
    assert "?" in n
    assert len(s) > len(n)
    print(f"[OK] English: {n} | {s[:50]}...")

async def test_bangla():
    n, s = await preprocess_query("নিউটনের সূত্র কি", "Bangla", gpt_4o_mini_complete, {})
    assert "নিউটন" in n
    print(f"[OK] Bangla: {n} | {s[:50]}...")

async def test_typo():
    n, s = await preprocess_query("whn was einstien born", "English", gpt_4o_mini_complete, {})
    assert "einstein" in n.lower() and "when" in n.lower()
    print(f"[OK] Typo: {n} | {s[:50]}...")

if __name__ == "__main__":
    asyncio.run(test_english())
    asyncio.run(test_bangla())
    asyncio.run(test_typo())
