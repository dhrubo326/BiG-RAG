"""
Quick Evaluation Script - Compare BiG-RAG with Baselines

Runs a fast evaluation comparing:
1. BiG-RAG (hybrid mode) - Full bipartite graph with entities + relations
2. BiG-RAG (local mode) - Entity-based retrieval only
3. BiG-RAG (global mode) - Relation-based retrieval only
4. BiG-RAG (naive mode) - Direct text similarity (no graph structure)

Usage:
    python script_quick_eval.py --sample 20  # Evaluate on 20 random samples
    python script_quick_eval.py --full       # Full evaluation (all questions)
"""

import asyncio
import argparse
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json

from script_evaluate_single_topic import SingleTopicEvaluator
from bigrag.base import QueryParam


class QuickComparativeEvaluator(SingleTopicEvaluator):
    """Fast comparative evaluation across retrieval modes"""

    async def quick_evaluate_mode(
        self,
        mode: str,
        sample_size: int = 20,
        top_k: int = 10
    ) -> Dict:
        """Quick evaluation on a sample of questions"""
        print(f"\n{'='*80}")
        print(f"Evaluating Mode: {mode.upper()}")
        print(f"{'='*80}")

        # Sample questions from each type
        if sample_size > 0:
            single_sample = self.single_passage_qa.sample(
                min(sample_size, len(self.single_passage_qa)),
                random_state=42
            )
            multi_sample = self.multi_passage_qa.sample(
                min(sample_size, len(self.multi_passage_qa)),
                random_state=42
            )
            no_answer_sample = self.no_answer_qa.sample(
                min(sample_size, len(self.no_answer_qa)),
                random_state=42
            )
        else:
            # Use all questions
            single_sample = self.single_passage_qa
            multi_sample = self.multi_passage_qa
            no_answer_sample = self.no_answer_qa

        all_metrics = {
            "relevance": [],
            "comprehensiveness": [],
            "diversity": [],
            "logicality": [],
            "coherence": []
        }

        # Evaluate each sample
        for df, question_type in [
            (single_sample, "single"),
            (multi_sample, "multi"),
            (no_answer_sample, "no_answer")
        ]:
            for idx, row in df.iterrows():
                question = row["question"]

                # Get ground truth
                if "document_index" in row:
                    ground_truth_docs = {int(row["document_index"])}
                else:
                    ground_truth_docs = set()

                # Retrieve with specific mode
                results = await self.bigrag.aquery(
                    query=question,
                    param=QueryParam(top_k=top_k, mode=mode)
                )

                # Extract doc indices
                all_retrieved_docs = self.extract_doc_indices_from_results(results)

                # Calculate metrics
                relevance = self.calculate_relevance(all_retrieved_docs, ground_truth_docs)
                comprehensiveness = self.calculate_comprehensiveness(all_retrieved_docs, ground_truth_docs)
                diversity = self.calculate_diversity(all_retrieved_docs, ground_truth_docs)
                logicality = self.calculate_logicality(all_retrieved_docs, ground_truth_docs, all_retrieved_docs)
                coherence = self.calculate_coherence(results, ground_truth_docs)

                all_metrics["relevance"].append(relevance["f1"])
                all_metrics["comprehensiveness"].append(comprehensiveness)
                all_metrics["diversity"].append(diversity)
                all_metrics["logicality"].append(logicality)
                all_metrics["coherence"].append(coherence)

        # Aggregate
        import numpy as np
        results = {
            "mode": mode,
            "sample_size": len(all_metrics["relevance"]),
            "metrics": {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
                for metric, values in all_metrics.items()
            }
        }

        return results

    async def run_comparative_evaluation(
        self,
        sample_size: int = 20,
        top_k: int = 10
    ) -> Dict:
        """Compare all retrieval modes"""
        print("\n" + "="*80)
        print("BiG-RAG Comparative Evaluation")
        print(f"Sample size: {sample_size} per question type")
        print(f"Top-k: {top_k}")
        print("="*80)

        # Build graph if needed
        if self.bigrag is None:
            await self.build_graph()

        # Test each mode
        modes = ["hybrid", "local", "global", "naive"]
        results = {}

        for mode in modes:
            results[mode] = await self.quick_evaluate_mode(
                mode=mode,
                sample_size=sample_size,
                top_k=top_k
            )

        # Save results
        output_file = self.working_dir / "comparative_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

        # Print comparison table
        self.print_comparison_table(results)

        return results

    def print_comparison_table(self, results: Dict):
        """Print comparison table"""
        print("\n" + "="*80)
        print("COMPARATIVE RESULTS")
        print("="*80)

        # Header
        print(f"\n{'Metric':<20} {'Hybrid':>12} {'Local':>12} {'Global':>12} {'Naive':>12}")
        print("-" * 80)

        # Each metric
        metrics = ["relevance", "comprehensiveness", "diversity", "logicality", "coherence"]
        for metric in metrics:
            values = [
                results[mode]["metrics"][metric]["mean"]
                for mode in ["hybrid", "local", "global", "naive"]
            ]

            # Find best
            best_idx = values.index(max(values))

            # Print row
            row = f"{metric.capitalize():<20}"
            for i, (mode, value) in enumerate(zip(["hybrid", "local", "global", "naive"], values)):
                if i == best_idx:
                    row += f" \033[1m{value:>11.3f}*\033[0m"  # Bold + asterisk for best
                else:
                    row += f" {value:>12.3f}"
            print(row)

        print("\n* = Best performance")

        # Statistical significance note
        print("\nNote: Run with --full for complete evaluation on all questions")

        # Recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)

        hybrid_score = sum(results["hybrid"]["metrics"][m]["mean"] for m in metrics) / len(metrics)
        best_baseline = max(
            ("local", sum(results["local"]["metrics"][m]["mean"] for m in metrics) / len(metrics)),
            ("global", sum(results["global"]["metrics"][m]["mean"] for m in metrics) / len(metrics)),
            ("naive", sum(results["naive"]["metrics"][m]["mean"] for m in metrics) / len(metrics)),
            key=lambda x: x[1]
        )

        improvement = ((hybrid_score - best_baseline[1]) / best_baseline[1]) * 100

        if improvement > 5:
            print(f"✓ BiG-RAG (hybrid) outperforms best baseline ({best_baseline[0]}) by {improvement:.1f}%")
            print("  → Bipartite graph structure provides clear value!")
        elif improvement > 0:
            print(f"⚠ BiG-RAG (hybrid) slightly outperforms best baseline ({best_baseline[0]}) by {improvement:.1f}%")
            print("  → Consider tuning parameters or using more complex queries")
        else:
            print(f"⚠ Best baseline ({best_baseline[0]}) outperforms BiG-RAG (hybrid) by {-improvement:.1f}%")
            print("  → Check entity extraction quality and graph construction")


async def main():
    parser = argparse.ArgumentParser(description="Quick comparative evaluation")
    parser.add_argument("--data_source", type=str, default="Single-Topic", help="Dataset name")
    parser.add_argument("--working_dir", type=str, default="../expr", help="Working directory")
    parser.add_argument("--sample", type=int, default=20, help="Sample size per question type")
    parser.add_argument("--full", action="store_true", help="Use all questions (no sampling)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild graph")

    args = parser.parse_args()

    # Load OpenAI API key (from parent directory)
    import os
    api_key_file = Path("../openai_api_key.txt")
    if api_key_file.exists():
        with open(api_key_file, 'r') as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
        print("[OK] Loaded OpenAI API key from openai_api_key.txt\n")
    elif "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OpenAI API key not found!")
        print("Please create openai_api_key.txt or set OPENAI_API_KEY environment variable")
        return

    sample_size = 0 if args.full else args.sample

    evaluator = QuickComparativeEvaluator(
        data_source=args.data_source,
        working_dir=args.working_dir
    )

    # Check if graph exists (check for vdb_entities.json which is created by new implementation)
    graph_exists = (Path(args.working_dir) / args.data_source / "vdb_entities.json").exists()

    if args.rebuild or not graph_exists:
        await evaluator.build_graph()
    else:
        print(f"\n[OK] Using existing graph at {Path(args.working_dir) / args.data_source}")
        from bigrag import BiGRAG
        evaluator.bigrag = BiGRAG(
            working_dir=str(Path(args.working_dir) / args.data_source),
            enable_llm_cache=True
        )

    # Run comparative evaluation
    results = await evaluator.run_comparative_evaluation(
        sample_size=sample_size,
        top_k=args.top_k
    )

    return results


if __name__ == "__main__":
    asyncio.run(main())
