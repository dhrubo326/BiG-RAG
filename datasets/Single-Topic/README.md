# Single-Topic Evaluation Dataset

Small, focused dataset for testing BiG-RAG end-to-end pipeline with OpenAI models.

## Quick Stats

- **20 documents** across diverse topics (gaming, tech, travel, fiction)
- **164 evaluation questions** across 3 types:
  - 63 single-passage questions (answer in 1 document)
  - 60 multi-passage questions (answer spans multiple sections)
  - 41 no-answer questions (unanswerable from corpus)

## Quick Start

```bash
# 1. Convert CSV to corpus
cd retrieval_eval
python convert_csv_to_corpus.py --csv ../datasets/Single-Topic/raw/documents.csv

# 2. Build knowledge graph (10-30 min)
cd ..
python script_build.py --data_source Single-Topic

# 3. Evaluate (5 min)
cd retrieval_eval
python script_quick_eval.py --data_source Single-Topic --sample 20
```

**See**: [SINGLE_TOPIC_EVALUATION.md](../../SINGLE_TOPIC_EVALUATION.md) for complete guide

## Files

### Corpus
- `raw/documents.csv` - 20 documents (index, source_url, text)
- `raw/corpus.jsonl` - Converted format for BiG-RAG (auto-generated)

### Evaluation Questions
- `raw/single_passage_answer_questions.csv` - 63 questions
- `raw/multi_passage_answer_questions.csv` - 60 questions
- `raw/no_answer_questions.csv` - 41 questions

## Document Topics

Gaming (6), Technical/AI (6), Fiction/Media (4), Other (4)

Examples: Enter the Gungeon, D&D logs, RAG systems, LLMWare, Marimo, EU AI Act, travel blog, short story, Doctor Who, etc.

## Evaluation Metrics

1. **Relevance (F1)**: Precision/recall of retrieved docs (target: >0.80)
2. **Comprehensiveness**: Coverage of all needed docs (target: >0.90)
3. **Diversity**: Multi-source retrieval (target: >0.85)
4. **Logicality**: Signal-to-noise ratio (target: >0.75)
5. **Coherence**: Ranking quality (target: >0.85)

## Expected Results

```
Metric               Hybrid      Local     Global      Naive
Relevance            0.768*      0.732     0.654      0.621
Comprehensiveness    0.845*      0.798     0.723      0.712
Diversity            0.892*      0.867     0.801      0.789

* = Best performance (hybrid mode uses full bipartite graph)
```

---

**Complete documentation**: [SINGLE_TOPIC_EVALUATION.md](../../SINGLE_TOPIC_EVALUATION.md)
