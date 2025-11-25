# Developer Brief: Enhanced Pipeline Redesign

**Date:** November 25, 2025
**Project:** BiG-RAG Knowledge Graph Construction
**Objective:** Redesign production pipeline to incorporate best practices and prepare for future unification

---

## Executive Summary

We're redesigning the production knowledge graph construction pipeline (renaming it to "Enhanced Pipeline") to address three critical gaps: (1) context loss during chunking, (2) poor entity recall from narrative text, and (3) code duplication between pipelines. The redesigned Enhanced Pipeline will combine the best features from both our standard and production pipelines - semantic boundary-aware chunking from standard pipeline's approach, gleaning-based extraction for better recall, and production's strict validation system - all while maintaining backward compatibility and preparing the codebase for eventual unification into a single pipeline implementation.

## Background & Problem Statement

We currently have two separate knowledge graph construction pipelines serving different use cases. The standard pipeline is fast and uses multi-pass gleaning extraction (finding 20-30% more entities through iterative refinement), but lacks validation for structured data like tables. The production pipeline has excellent table handling and numeric validation but uses single-pass extraction that misses entities in complex narrative paragraphs. Both pipelines use fixed token-window chunking that blindly splits text at token boundaries, causing critical context loss - for example, when testing on KUET admission documents, the query "প্রথম ২০,০০০ প্রার্থী কিভাবে নির্বাচন করা হবে?" failed to retrieve the complete selection criteria because the paragraph was split mid-context across two chunks. Additionally, both pipelines have duplicated entity merging logic with different implementations, making maintenance difficult.

## Technical Analysis

**Chunking Workflow Clarification:** Both pipelines currently use batch processing (Option B), which is the industry standard used by LangChain, LlamaIndex, and other production RAG systems. The workflow is: (1) chunk ALL documents first into a complete set of chunks, (2) extract entities from ALL chunks in batch using concurrent async calls, (3) merge entities with the same name across chunks (aggregating weights and source IDs), (4) insert everything to graph and vector stores in bulk. This batch approach enables cross-chunk entity merging, which is critical for graph quality - for example, if "CSE" is mentioned in three different chunks, we create a single entity node with weight aggregated from all occurrences rather than three duplicate nodes. We do NOT create cross-chunk relations (relations between entities from different chunks) because that would require inference beyond the source text and risks hallucination - relations are only created within each chunk during extraction, and cross-document connections emerge naturally through shared entity nodes.

**Current Standard Pipeline Strengths:** Uses gleaning extraction with 3 LLM passes (initial extraction + 2 gleaning passes), where each gleaning pass sees the conversation history and is prompted to find missed entities, then uses quality scoring to pick the best description when the same entity is found multiple times. This produces 20-30% higher entity recall compared to single-pass extraction, especially important for narrative paragraphs where entities may be mentioned indirectly. Entity merging is simple name-based grouping with weight aggregation (fast, effective for most cases).

**Current Production Pipeline Strengths:** Uses table-aware chunking that extracts tables first and converts them to natural language for embedding, ensuring table data is never split. Applies strict numeric validation where all numbers from source text must appear in extracted entities/relations with zero hallucination tolerance. Uses advanced entity canonicalization with fuzzy string matching (Levenshtein distance) and embedding similarity to merge entity variants like "CSE", "C.S.E.", "Computer Science Engineering" into a single entity. However, paragraph extraction uses single-pass validation retry that doesn't learn from previous attempts - if extraction fails validation, it retries with the SAME prompt rather than gleaning with conversation history.

**Critical Gap Identified:** Production pipeline lacks gleaning for paragraphs. When testing on KUET dataset, production pipeline extracted 12 contexts for "বায়োমেডিকেল ইঞ্জিনিয়ারিং বিভাগে আসন সংখ্যা কত?" versus standard's 18 contexts - the missing 6 contexts contained entities that single-pass extraction missed but gleaning found in subsequent passes. This is because production's "retry" is just error recovery (same prompt, no learning), not true gleaning (conversation history, iterative refinement).

## Solution: Enhanced Pipeline Redesign

We will redesign the production pipeline (renaming it to "Enhanced Pipeline") through 5 implementation steps over 3 weeks. The core philosophy is to merge the best of both worlds - standard's gleaning recall + production's validation accuracy - while maintaining code that both pipelines can eventually share for future unification.

### Step 1: Extraction Strategy Configuration (Week 1, 4 hours)

Add a configurable extraction strategy parameter that controls how entity extraction is performed. Users will be able to choose from three strategies: "strict" for single-pass extraction with validation (fastest, use for tables and structured data with 95%+ accuracy), "gleaning" for multi-pass extraction with validation (slowest but 98%+ accuracy, best for narrative paragraphs), or "hybrid" for adaptive extraction that automatically uses strict mode for tables and gleaning mode for paragraphs (recommended default, balances speed and accuracy). Implementation involves adding new parameters to EnhancedKGPipeline.__init__() (renamed from ProductionKGPipeline) and ConstrainedLLMExtractor class: enable_gleaning (boolean), max_gleaning_iterations (default 2), and extraction_strategy ('strict' | 'gleaning' | 'hybrid'). The enhanced pipeline config in bigrag.py will change from use_production_pipeline to use_enhanced_pipeline with new config key extraction_strategy defaulting to "hybrid". Backward compatibility is maintained by keeping the old config key with a deprecation warning. Success criteria: user can set extraction strategy in config, hybrid mode correctly routes tables to strict and paragraphs to gleaning, all existing tests pass.

### Step 2: Semantic Boundary-Aware Chunking (Week 1-2, 10-12 hours)

Implement smart paragraph chunking that respects semantic boundaries instead of blindly splitting at token positions. Current behavior: both pipelines use fixed sliding window that splits at token 1200 regardless of content, causing the "20,000 candidate selection criteria" paragraph to split mid-sentence across chunks. New behavior: detect paragraph boundaries (double newlines), keep paragraphs under 1300 tokens intact (30% overflow tolerance for completeness), split only at sentence boundaries for larger paragraphs, and use 200 tokens overlap (100 before + 100 after) composed of complete sentences. The algorithm works as follows: (1) split document by double newlines into paragraphs, (2) for each paragraph, if it's less than 1300 tokens, keep the entire paragraph intact and accumulate with other small paragraphs until reaching chunk_size, (3) if a paragraph exceeds 1300 tokens, split it at sentence boundaries (handling both Bengali "।" and English ".!?"), (4) when creating chunks, add 100 tokens overlap before (last few sentences from previous chunk) and 100 tokens overlap after (first few sentences from next chunk), ensuring overlap consists of complete sentences only. Implementation adds new method _chunk_with_semantic_boundaries() to TableAwareChunker class and new utility functions to bigrag/utils.py: count_tokens_fast() for approximate token counting (4 chars ≈ 1 token), split_by_sentences() that handles Bengali and English sentence endings. Tables continue to use existing behavior (never split, each table is one chunk). Testing requirement: verify that the KUET selection criteria paragraph stays intact in one chunk, Query "প্রথম ২০,০০০ প্রার্থী কিভাবে নির্বাচন করা হবে?" must retrieve complete answer. Expected impact: +20% context retention, fixes missing Q5 answer issue, improves retrieval precision by 10-15%.

### Step 3: Gleaning Implementation in Enhanced Pipeline (Week 2, 8-10 hours)

Add multi-pass gleaning loop to ConstrainedLLMExtractor that is IDENTICAL to standard pipeline's gleaning logic, enabling future code unification. Current production behavior: extract_from_paragraph() does single extraction with validation retry up to 3 attempts, but each retry uses the same prompt without conversation history (stateless error recovery). New behavior: after initial extraction passes validation, if gleaning is enabled, perform 2 additional gleaning passes where LLM sees full conversation history and is prompted "CONTINUE EXTRACTION: Review the text again and identify ANY additional entities you may have missed", then merge gleaned entities using quality-based comparison - for entities with the same name, keep the version with higher description_quality_score() (scoring factors: length up to 40 points, keyword density up to 30 points, specificity indicators like numbers/dates up to 30 points), for new entities not in original extraction, add them to the merged result. Implementation refactors extract_from_paragraph() to call new method _extract_once() for single-pass extraction, then adds gleaning loop that builds conversation_history array with user/assistant messages, calls LLM with conversation context, validates each gleaning result, and merges using new method _merge_extractions_by_quality() that imports description_quality_score from bigrag/utils.py (same function standard pipeline uses). The gleaning prompt must be IDENTICAL to standard pipeline's continue_prompt so future unification is seamless. Configuration: enable_gleaning and max_gleaning_iterations are constructor parameters, with hybrid mode setting enable_gleaning=True for paragraph chunks and False for table chunks. Success criteria: gleaning can be toggled via config, finds 20-30% more entities for narrative paragraphs, quality-based merging produces same results as standard pipeline's smart merge, validation still applied after gleaning completes.

### Step 4: Unified Entity Merging (Week 3, 6 hours)

Extract entity merging logic into a standalone module that both pipelines can import, eliminating code duplication. Current state: standard pipeline has inline merging in operate.py::_merge_nodes_then_upsert() that groups entities by name (case-insensitive) and aggregates weights/source_ids (simple, fast), enhanced pipeline has separate implementation using EntityCanonicalizationMap and SimpleEntityLinker with fuzzy matching and embedding similarity (complex, accurate). New approach: create bigrag/merging/unified_merger.py with UnifiedEntityMerger class that supports two strategies: "basic" for name-based grouping (replicates standard pipeline logic) and "fuzzy" for canonicalization + fuzzy matching (replicates enhanced pipeline logic). Implementation: UnifiedEntityMerger.__init__() takes strategy parameter and fuzzy_threshold (default 0.90), merge_entities() method accepts list of raw entities and returns merged list, _merge_basic() groups by normalized name and aggregates weights/descriptions/source_ids (identical to standard's current logic), _merge_fuzzy() delegates to existing SimpleEntityLinker (reuses enhanced pipeline's current implementation). Integration: both pipelines import UnifiedEntityMerger, standard pipeline uses strategy='basic' by default (can opt into fuzzy via addon_params), enhanced pipeline uses strategy='fuzzy' if enable_entity_linking=True else 'basic'. Success criteria: both pipelines produce identical results to their current implementations, no duplicated merging code, tests verify basic strategy matches old standard behavior and fuzzy strategy matches old enhanced behavior.

### Step 5: Pipeline Selection Helper (Week 3, 3 hours)

Add optional helper function that recommends extraction strategy based on document characteristics. Implementation: create bigrag/pipeline_selector.py with recommend_extraction_strategy() function that analyzes document text and returns 'strict', 'gleaning', or 'hybrid' based on heuristics: if document has tables (markdown table syntax detected) AND has many numbers AND word count under 3000 → return 'strict' (structured data, fast extraction sufficient), if word count over 5000 → return 'gleaning' (long narrative, need thorough extraction), else return 'hybrid' (mixed content, adaptive approach). This is a convenience function - users can still manually set extraction_strategy in config. Success criteria: recommendations are sensible for test documents, function is documented as optional helper.

## Technical Specifications

**Naming Convention:** ProductionKGPipeline class renamed to EnhancedKGPipeline, use_production_pipeline config key renamed to use_enhanced_pipeline (with deprecation warning for old key), production_pipeline_config renamed to enhanced_pipeline_config. File bigrag/production_pipeline.py renamed to bigrag/enhanced_pipeline.py. This naming better reflects purpose (enhanced accuracy/validation) rather than implying production-readiness.

**Configuration Schema:**
```python
# In bigrag/bigrag.py
use_enhanced_pipeline: bool = False  # Default to standard for backward compatibility
enhanced_pipeline_config: dict = {
    "validation_level": "MODERATE",  # STRICT | MODERATE | LENIENT
    "enable_entity_linking": True,
    "extraction_strategy": "hybrid"  # strict | gleaning | hybrid
}
```

**Chunk Size Parameters:** Max chunk size reduced from 1200 to 1000 tokens (more conservative), overlap increased from 100 to 200 total tokens (100 before + 100 after chunk), tolerance for complete paragraphs set to 1300 tokens (30% overflow allowance). Token counting uses approximation of 4 characters per token for performance (can be replaced with proper tiktoken in future if needed).

**Gleaning Configuration:** max_gleaning_iterations defaults to 2 (same as standard pipeline's entity_extract_max_gleaning), gleaning prompt format must match standard pipeline's PROMPTS["entiti_continue_extraction"] for future unification, quality scoring uses description_quality_score() from bigrag/utils.py (shared function between pipelines), merged extraction includes metadata field 'extraction_method' with value 'constrained_llm_with_gleaning' to track which method was used.

**Entity Merging Strategies:**
- Basic strategy: groups by entity_name.strip().lower(), sums weights from all occurrences, collects all source_ids with GRAPH_FIELD_SEP separator, picks longest description, generates stable entity_id using compute_mdhash_id(entity_name, prefix=ENTITY_PREFIX)
- Fuzzy strategy: applies EntityCanonicalizationMap for predefined aliases (e.g., CSE → Computer Science and Engineering), uses Levenshtein distance > fuzzy_threshold (default 0.90) for string matching, computes embedding similarity > 0.85 for semantic matching, merges matched variants into single primary entity with entity_ids_merged list tracking all variant IDs

**Testing Strategy:** Unit tests for each component (test_smart_chunking.py verifies boundary preservation, test_gleaning.py verifies recall improvement, test_unified_merger.py verifies strategy correctness), integration test test_enhanced_pipeline_e2e.py runs full document through pipeline and compares output structure with standard pipeline, regression tests verify all existing KUET dataset tests pass with new implementation, backward compatibility test verifies old config keys still work with deprecation warnings.

## Implementation Guidelines

**Code Organization:** Keep all changes within enhanced pipeline - do NOT modify standard pipeline code (except adding imports for shared components like UnifiedEntityMerger). When adding gleaning to ConstrainedLLMExtractor, ensure the implementation is pluggable (can be disabled) so enhanced pipeline remains usable for strict-mode-only scenarios. All new utility functions (count_tokens_fast, split_by_sentences, description_quality_score) should be added to bigrag/utils.py as standalone functions that both pipelines can import. File renames should be done carefully with git mv to preserve history.

**Backward Compatibility:** Old config key use_production_pipeline must continue to work with deprecation warning pointing users to new key use_enhanced_pipeline. All existing tests must pass without modification - if tests fail, fix the implementation not the tests. Public API surface should not break - bigrag.py module-level exports remain unchanged. When renaming classes/files, consider providing deprecated aliases for one release cycle.

**Performance Considerations:** Gleaning adds ~2x LLM calls for paragraphs (1 initial + 2 gleaning passes), so hybrid mode is recommended default rather than full gleaning. Semantic chunking adds minimal overhead (regex splits and token counting approximation are fast operations). Entity merging strategies should be benchmarked - basic strategy is O(n) while fuzzy strategy is O(n²) due to pairwise comparisons, so fuzzy should only be used when accuracy is critical and n < 1000 entities.

**Error Handling:** Gleaning passes should fail gracefully - if gleaning pass fails (LLM error, validation failure), log warning and continue with previous extraction result rather than failing entire document. Semantic chunking should have fallback to fixed chunking if paragraph detection fails (malformed markdown). Entity merging should handle missing fields gracefully with sensible defaults (e.g., weight=0 if not provided).

**Logging and Debugging:** Add INFO-level logging for strategy selection (e.g., "[STRATEGY] Using gleaning for paragraph chunks, strict for tables"), DEBUG-level logging for gleaning results (e.g., "[GLEANING] Pass 2/2: Found 3 new entities, improved 2 descriptions"), log quality scores during merging (e.g., "[MERGE] Entity 'CSE': Gleaned version better (quality 45 → 78)"). Each major step should log progress for long-running operations.

## Migration and Rollout Plan

**Phase 1 (Current):** Complete Enhanced Pipeline redesign over 3 weeks following this plan. Enhanced pipeline remains opt-in (use_enhanced_pipeline=False by default). Standard pipeline remains unchanged and continues to be default for all users.

**Phase 2 (Future, 1-2 months after Phase 1):** Add toggle to standard pipeline configuration: use_enhanced_components=True that makes standard pipeline use UnifiedEntityMerger and smart chunking from enhanced pipeline. Document hybrid approach in user guide. Collect feedback from early adopters using enhanced pipeline. Run benchmark comparisons on multiple datasets (KUET, 2WikiMultiHopQA, HotpotQA) to validate quality improvements.

**Phase 3 (Future, 3-4 months after Phase 1):** Gradually deprecate standard pipeline's old implementations by making use_enhanced_components=True the default. Monitor for issues, provide fallback option. Eventually delete redundant code after sufficient soak time.

**Phase 4 (Future, 6+ months):** Full unification - rename EnhancedKGPipeline to UnifiedKGPipeline, remove all legacy pipeline code, single implementation with strategy selection. This is the end goal but requires careful migration to avoid breaking existing users.

## Expected Outcomes

**Quantitative Improvements:** Paragraph entity recall increases from 70-80% to 90-95% (+15-20% absolute gain), context completeness for retrieval increases from 75% to 95% (+20%, fixes Q5 missing answer issue), entity deduplication improves from 60% to 90% (+30%, fewer duplicate nodes), overall F1 score on KUET dataset improves from 82% to 90-92% (+8-10 points). Trade-off: extraction time for paragraphs increases by ~2x due to gleaning passes (acceptable given quality gains).

**Qualitative Improvements:** Code maintainability significantly improved through unification (single entity merging implementation instead of two), future pipeline consolidation path is clear (both pipelines can use same components), user experience improved with configurable strategies (can choose speed vs accuracy trade-off), better debugging and monitoring through structured logging.

**Testing Success Criteria:** Query "প্রথম ২০,০০০ প্রার্থী কিভাবে নির্বাচন করা হবে?" retrieves complete answer in single chunk (currently fails), query "CSE তে কত আসন আছে?" returns fewer than 10 contexts (currently over-retrieves 18-20), all existing unit tests pass without modification, benchmark F1 score on KUET dataset shows 8-10 point improvement.

## Timeline and Milestones

**Week 1:** Complete Step 1 (extraction strategy config, 4 hours) and begin Step 2 (semantic chunking, 10-12 hours). Deliverable: extraction_strategy parameter works correctly, semantic chunking passes unit tests. Milestone: hybrid mode can route different chunk types to different extractors.

**Week 2:** Complete Step 2 (semantic chunking) and Step 3 (gleaning implementation, 8-10 hours). Deliverable: Q5 retrieval test passes (complete answer retrieved), gleaning finds 20-30% more entities than single-pass. Milestone: enhanced pipeline has feature parity with standard for narrative extraction.

**Week 3:** Complete Step 4 (unified merger, 6 hours) and Step 5 (selection helper, 3 hours), comprehensive testing and documentation. Deliverable: UnifiedEntityMerger works in both pipelines, all tests pass, updated documentation. Milestone: enhanced pipeline is production-ready and well-tested.

## Questions and Clarifications

If you have questions during implementation, refer to the detailed plan document (Production_pipeline_redesign_plan.md) which contains code examples and technical specifications for each step. For architectural decisions, the key points are: (1) batch processing is correct approach for chunking workflow, (2) no cross-chunk relation creation (only entity merging), (3) gleaning logic must be identical to standard pipeline for future unification, (4) semantic chunking has 30% overflow tolerance to keep paragraphs intact. If you encounter unexpected issues or need to deviate from the plan, document the reason and proposed alternative approach for team review before proceeding.

---

**Contact:** For questions or clarifications, reach out via project communication channels. This brief should be read alongside Production_pipeline_redesign_plan.md for complete technical specifications.
