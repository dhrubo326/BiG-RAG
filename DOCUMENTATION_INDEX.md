# BiG-RAG Documentation Index

**Complete guide to BiG-RAG documentation and resources**

---

## 📚 Core Documentation

### Getting Started

1. **[README.md](README.md)** - Project overview and quick start
   - Introduction to BiG-RAG
   - Installation instructions (venv and conda)
   - Quick start guide for 2WikiMultiHopQA
   - Training and evaluation basics

2. **[CLAUDE.md](CLAUDE.md)** - Complete developer reference
   - Detailed architecture overview
   - Data pipeline workflow
   - Configuration system
   - Common commands and gotchas
   - Debugging and logging

3. **[SETUP_VENV.md](SETUP_VENV.md)** - Virtual environment setup
   - Step-by-step venv installation
   - Dependency management
   - BiG-RAG-only mode (no RL training)

---

## 📊 Dataset and Corpus

4. **[docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md)** ⭐ **NEW**
   - **What is a corpus?** Complete explanation
   - **Dataset structure** and file formats
   - **Corpus construction** for different dataset types
   - **Complete data pipeline** walkthrough
   - **Building custom datasets** step-by-step
   - **Extending existing datasets**
   - **Troubleshooting** common issues

---

## 🔄 Rebranding Documentation

5. **[REBRANDING_COMPLETION_SUMMARY.md](REBRANDING_COMPLETION_SUMMARY.md)** ⭐ **START HERE**
   - **Complete rebranding overview**
   - All phases and changes summarized
   - Migration guide and verification status
   - Backward compatibility details

6. **[REBRANDING_AUDIT_2025.md](REBRANDING_AUDIT_2025.md)** ⭐ **DETAILED AUDIT**
   - Deep examination of entire codebase
   - All 12 files updated in final audit
   - Before/after code comparisons
   - Testing recommendations

7. **[REBRANDING_SUMMARY.md](REBRANDING_SUMMARY.md)** - Quick rebranding overview
   - What changed (Graph-R1 → BiG-RAG)
   - Key naming changes
   - Migration guide for users

8. **[REBRANDING_CHANGELOG.md](REBRANDING_CHANGELOG.md)** - Detailed change log
   - Complete list of modified files
   - Before/after comparisons
   - Verification checklist

9. **[REBRANDING_PLAN.md](REBRANDING_PLAN.md)** - Strategic rebranding plan
   - Rationale for rebranding
   - Complete checklist of changes
   - Implementation order

10. **[FILE_RENAME_UPDATE.md](FILE_RENAME_UPDATE.md)** - File rename details
    - `graphr1.py` → `bigrag.py` rename
    - Import updates across all files
    - Verification results

---

## 🏗️ Architecture and Deep Dives

11. **[docs/GRAPH_R1_VS_BIGRAG_DEEP_DIVE.md](docs/GRAPH_R1_VS_BIGRAG_DEEP_DIVE.md)**
    - Comparison between Graph-R1 and BiG-RAG
    - Architectural differences

12. **[docs/DEEP_DIVE_INDEXING_PIPELINES.md](docs/DEEP_DIVE_INDEXING_PIPELINES.md)**
    - Detailed indexing pipeline explanation
    - FAISS index construction

13. **[docs/BIPARTITE_VALIDATION_AND_RELEVANCE_FILTERING_COMPARISON.md](docs/BIPARTITE_VALIDATION_AND_RELEVANCE_FILTERING_COMPARISON.md)**
    - Bipartite graph validation
    - Relevance filtering techniques

14. **[docs/EDUCATIONAL_DEEP_DIVE_RETRIEVAL_ARCHITECTURES.md](docs/EDUCATIONAL_DEEP_DIVE_RETRIEVAL_ARCHITECTURES.md)**
    - Educational overview of retrieval architectures
    - RAG system comparisons

15. **[docs/SETUP_AND_TESTING_GUIDE.md](docs/SETUP_AND_TESTING_GUIDE.md)**
    - Complete setup instructions
    - Testing procedures

---

## 📖 Original Research

16. **[docs/Graph-R1_full_paper.md](docs/Graph-R1_full_paper.md)**
    - Original Graph-R1 paper (full text)
    - Kept as historical reference

17. **[docs/BiG-RAG_Full_Paper.md](docs/BiG-RAG_Full_Paper.md)**
    - BiG-RAG paper with updated terminology

---

## 🔧 Helper Code and Examples

18. **[docs/Helper_code/README.md](docs/Helper_code/README.md)**
    - Helper scripts overview

19. **[docs/Helper_code/api_server.py](docs/Helper_code/api_server.py)**
    - Enhanced API server with multi-provider LLM support
    - Uses OpenAI embeddings

20. **[docs/Helper_code/build_knowledge_graph.py](docs/Helper_code/build_knowledge_graph.py)**
    - Knowledge graph construction with OpenAI embeddings
    - Alternative to script_build.py

---

## 🎯 Evaluation and Inference

21. **[evaluation/README.md](evaluation/README.md)**
    - Evaluation procedures
    - Metrics (EM, F1, etc.)

22. **[inference/README.md](inference/README.md)**
    - Inference guide
    - Loading trained models

---

## 📋 Quick Reference

### File Locations Cheatsheet

```
BiG-RAG/
├── README.md                                    # Start here
├── CLAUDE.md                                    # Developer guide
├── SETUP_VENV.md                               # Installation
├── docs/
│   ├── DATASET_AND_CORPUS_GUIDE.md            # ⭐ Dataset guide
│   ├── SETUP_AND_TESTING_GUIDE.md
│   ├── DEEP_DIVE_*.md                          # Architecture docs
│   ├── Graph-R1_full_paper.md                  # Original paper
│   ├── BiG-RAG_Full_Paper.md                   # Updated paper
│   └── Helper_code/                            # Example scripts
├── REBRANDING_COMPLETION_SUMMARY.md            # ⭐ Rebranding overview
├── REBRANDING_AUDIT_2025.md                    # ⭐ Detailed audit
├── REBRANDING_*.md                             # Other rebranding docs
├── FILE_RENAME_UPDATE.md                       # File rename log
└── DOCUMENTATION_INDEX.md                      # This file
```

### Documentation by Topic

#### **Installation & Setup**
- [README.md](README.md) - Quick start
- [SETUP_VENV.md](SETUP_VENV.md) - Virtual environment
- [docs/SETUP_AND_TESTING_GUIDE.md](docs/SETUP_AND_TESTING_GUIDE.md) - Complete setup

#### **Understanding BiG-RAG**
- [CLAUDE.md](CLAUDE.md) - Architecture overview
- [docs/GRAPH_R1_VS_BIGRAG_DEEP_DIVE.md](docs/GRAPH_R1_VS_BIGRAG_DEEP_DIVE.md) - Comparison
- [docs/EDUCATIONAL_DEEP_DIVE_RETRIEVAL_ARCHITECTURES.md](docs/EDUCATIONAL_DEEP_DIVE_RETRIEVAL_ARCHITECTURES.md) - RAG architectures

#### **Datasets & Corpus**
- **[docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md)** ⭐ - Complete guide
  - What is a corpus
  - Building datasets
  - Data pipeline
  - Troubleshooting

#### **Development**
- [CLAUDE.md](CLAUDE.md) - Developer reference
- [docs/Helper_code/](docs/Helper_code/) - Example code

#### **Training & Evaluation**
- [README.md](README.md) - Training basics
- [evaluation/README.md](evaluation/README.md) - Evaluation guide
- [inference/README.md](inference/README.md) - Inference guide

#### **Rebranding Information**
- [REBRANDING_SUMMARY.md](REBRANDING_SUMMARY.md) - Quick overview
- [REBRANDING_CHANGELOG.md](REBRANDING_CHANGELOG.md) - Detailed changes
- [FILE_RENAME_UPDATE.md](FILE_RENAME_UPDATE.md) - File rename details

---

## 🚀 Recommended Reading Order

### For New Users

1. Start with [README.md](README.md) - Get familiar with the project
2. Read [CLAUDE.md](CLAUDE.md) sections:
   - Project Overview
   - Environment Setup
   - Common Commands
3. Read [docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md) - Understand data
4. Follow quick start in [README.md](README.md)

### For Developers

1. Read [CLAUDE.md](CLAUDE.md) completely - Understand architecture
2. Read [docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md) - Data pipeline
3. Review [docs/Helper_code/](docs/Helper_code/) - Example implementations
4. Check [docs/DEEP_DIVE_INDEXING_PIPELINES.md](docs/DEEP_DIVE_INDEXING_PIPELINES.md) - Technical details

### For Researchers

1. Read [docs/Graph-R1_full_paper.md](docs/Graph-R1_full_paper.md) - Original paper
2. Read [docs/BiG-RAG_Full_Paper.md](docs/BiG-RAG_Full_Paper.md) - Updated paper
3. Read [REBRANDING_CHANGELOG.md](REBRANDING_CHANGELOG.md) - Understand changes
4. Review [docs/GRAPH_R1_VS_BIGRAG_DEEP_DIVE.md](docs/GRAPH_R1_VS_BIGRAG_DEEP_DIVE.md) - Comparison

### For Those Building Custom Datasets

1. **[docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md)** ⭐ - Start here!
   - Complete walkthrough of corpus construction
   - Step-by-step dataset building
   - Extending existing datasets
2. Review example datasets in `datasets/` directory
3. Check [docs/Helper_code/build_knowledge_graph.py](docs/Helper_code/build_knowledge_graph.py) - Example code

---

## 💡 Key Concepts Explained

### What is BiG-RAG?
BiG-RAG (Bipartite Graph Retrieval-Augmented Generation) trains LLMs to actively query knowledge graphs during generation using reinforcement learning.

### What is a Corpus?
A collection of text documents that forms the knowledge base. See [docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md) for complete explanation.

### What is a Bipartite Graph?
A graph with two types of nodes (documents and semantic entities/relations) where edges only connect nodes of different types. See [CLAUDE.md](CLAUDE.md) for visualization.

### Graph-R1 vs BiG-RAG?
BiG-RAG is the rebranded implementation with accurate terminology (bipartite graph instead of hypergraph). See [REBRANDING_SUMMARY.md](REBRANDING_SUMMARY.md).

---

## 🔍 Finding Information

### By Keyword

**Installation**: [README.md](README.md), [SETUP_VENV.md](SETUP_VENV.md), [CLAUDE.md](CLAUDE.md)

**Corpus**: **[docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md)** ⭐

**Dataset**: **[docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md)** ⭐, [README.md](README.md)

**Training**: [README.md](README.md), [CLAUDE.md](CLAUDE.md)

**Evaluation**: [evaluation/README.md](evaluation/README.md)

**Troubleshooting**: [CLAUDE.md](CLAUDE.md), [docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md)

**API**: [docs/Helper_code/api_server.py](docs/Helper_code/api_server.py)

**Rebranding**: [REBRANDING_SUMMARY.md](REBRANDING_SUMMARY.md), [REBRANDING_CHANGELOG.md](REBRANDING_CHANGELOG.md)

**File Rename**: [FILE_RENAME_UPDATE.md](FILE_RENAME_UPDATE.md)

---

## 📞 Getting Help

1. **Common issues?** Check [CLAUDE.md](CLAUDE.md) "Common Gotchas" section
2. **Dataset questions?** See [docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md) "Troubleshooting"
3. **Rebranding confusion?** Read [REBRANDING_SUMMARY.md](REBRANDING_SUMMARY.md)
4. **Still stuck?** Open an issue on GitHub

---

## 🆕 Recently Added

- **[docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md)** - Comprehensive dataset and corpus documentation
- **[FILE_RENAME_UPDATE.md](FILE_RENAME_UPDATE.md)** - Documentation of graphr1.py → bigrag.py rename
- **Updated [CLAUDE.md](CLAUDE.md)** - Fully updated with BiG-RAG terminology and current state

---

## 📝 Documentation Maintenance

**Last Updated**: 2025-10-24
**Version**: BiG-RAG 1.0.0 (post-rebranding)
**Status**: Complete and up-to-date

All documentation has been updated to reflect:
- ✅ BiG-RAG branding (from Graph-R1)
- ✅ Bipartite graph terminology (from hypergraph)
- ✅ File rename: bigrag.py (from graphr1.py)
- ✅ Updated imports and references
- ✅ Current project state

---

**Ready to get started?** Begin with [README.md](README.md) or jump to [docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md) to learn about datasets! 🚀
