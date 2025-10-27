"""
Comprehensive Evaluation Script for BiG-RAG on Single-Topic Dataset

Evaluates on 5 key metrics:
1. Relevance - Are retrieved chunks relevant to the query?
2. Comprehensiveness - Does it retrieve all necessary information?
3. Diversity - Does it retrieve from diverse sources when needed?
4. Logicality - Is the retrieval reasoning sound?
5. Coherence - Do retrieved chunks work together coherently?
"""

import pandas as pd
import numpy as np
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter
import argparse

# BiG-RAG imports
from bigrag import BiGRAG
from bigrag.utils import compute_args_hash
from bigrag.base import QueryParam


class SingleTopicEvaluator:
    """Evaluator for BiG-RAG on Single-Topic dataset"""

    def __init__(self, data_source: str = "Single-Topic", working_dir: str = "../expr"):
        self.data_source = data_source
        self.working_dir = Path(working_dir) / data_source
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset (relative to parent directory)
        self.raw_dir = Path("../datasets") / data_source / "raw"

        print("\n" + "="*80)
        print("LOADING DATASET FILES")
        print("="*80)

        # Load documents (for graph building - NOT used in evaluation directly)
        documents_path = self.raw_dir / "documents.csv"
        print(f"\n1. Corpus (for graph building):")
        print(f"   File: {documents_path}")
        if not documents_path.exists():
            raise FileNotFoundError(f"Documents file not found: {documents_path}")
        self.documents = pd.read_csv(documents_path)
        print(f"   ✓ Loaded {len(self.documents)} documents")
        print(f"   Purpose: Used to build knowledge graph (Step 2)")

        # Load question files (for evaluation - Step 3)
        print(f"\n2. Evaluation Questions (for testing retrieval):")

        single_path = self.raw_dir / "single_passage_answer_questions.csv"
        print(f"\n   A. Single-Passage Questions:")
        print(f"      File: {single_path}")
        if not single_path.exists():
            raise FileNotFoundError(f"File not found: {single_path}")
        self.single_passage_qa = pd.read_csv(single_path)
        print(f"      ✓ {len(self.single_passage_qa)} questions")
        print(f"      Purpose: Test basic retrieval (answer in 1 document)")

        multi_path = self.raw_dir / "multi_passage_answer_questions.csv"
        print(f"\n   B. Multi-Passage Questions:")
        print(f"      File: {multi_path}")
        if not multi_path.exists():
            raise FileNotFoundError(f"File not found: {multi_path}")
        self.multi_passage_qa = pd.read_csv(multi_path)
        print(f"      ✓ {len(self.multi_passage_qa)} questions")
        print(f"      Purpose: Test comprehensiveness (answer spans multiple docs/sections)")

        no_answer_path = self.raw_dir / "no_answer_questions.csv"
        print(f"\n   C. No-Answer Questions:")
        print(f"      File: {no_answer_path}")
        if not no_answer_path.exists():
            raise FileNotFoundError(f"File not found: {no_answer_path}")
        self.no_answer_qa = pd.read_csv(no_answer_path)
        print(f"      ✓ {len(self.no_answer_qa)} questions")
        print(f"      Purpose: Test false-positive avoidance (no answer exists)")

        total_questions = len(self.single_passage_qa) + len(self.multi_passage_qa) + len(self.no_answer_qa)
        print(f"\n   TOTAL: {total_questions} evaluation questions")
        print("="*80)

        # Initialize BiG-RAG
        self.bigrag = None

    async def build_graph(self):
        """Build BiG-RAG graph from documents"""
        print("\n" + "="*80)
        print("Building BiG-RAG Graph")
        print("="*80)

        # Initialize BiG-RAG
        self.bigrag = BiGRAG(
            working_dir=str(self.working_dir),
            enable_llm_cache=True
        )

        # Prepare documents in BiG-RAG format (ainsert expects list of strings)
        documents_for_insert = []
        for idx, row in self.documents.iterrows():
            documents_for_insert.append(row["text"])

        # Insert documents
        print(f"Inserting {len(documents_for_insert)} documents into BiG-RAG...")
        await self.bigrag.ainsert(documents_for_insert)
        print("[OK] Graph built successfully!")

    async def retrieve_for_query(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid"
    ) -> List[Dict]:
        """Retrieve relevant chunks for a query"""
        if self.bigrag is None:
            raise ValueError("Graph not built yet! Call build_graph() first.")

        # Query BiG-RAG
        results = await self.bigrag.aquery(
            query=query,
            param=QueryParam(top_k=top_k, mode=mode)
        )

        return results

    def extract_doc_indices_from_results(self, results: List[Dict]) -> Set[int]:
        """Extract document indices from retrieval results"""
        doc_indices = set()
        for result in results:
            # Results contain chunks with source document IDs
            doc_id = result.get("id", "").split("_")[0]  # Format: "docid_chunkid"
            try:
                doc_indices.add(int(doc_id))
            except (ValueError, AttributeError):
                pass
        return doc_indices

    def calculate_relevance(
        self,
        retrieved_docs: Set[int],
        ground_truth_docs: Set[int]
    ) -> Dict[str, float]:
        """
        Calculate relevance metrics
        - Precision: % of retrieved docs that are relevant
        - Recall: % of relevant docs that are retrieved
        - F1: Harmonic mean
        """
        if len(retrieved_docs) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if len(ground_truth_docs) == 0:
            # For no-answer questions, precision should be low if docs retrieved
            return {
                "precision": 0.0,
                "recall": 1.0 if len(retrieved_docs) == 0 else 0.0,
                "f1": 0.0
            }

        true_positives = len(retrieved_docs & ground_truth_docs)
        precision = true_positives / len(retrieved_docs) if retrieved_docs else 0.0
        recall = true_positives / len(ground_truth_docs) if ground_truth_docs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def calculate_comprehensiveness(
        self,
        retrieved_docs: Set[int],
        ground_truth_docs: Set[int]
    ) -> float:
        """
        Calculate comprehensiveness: Did we retrieve ALL necessary documents?
        Returns recall (what % of ground truth docs were retrieved)
        """
        if len(ground_truth_docs) == 0:
            return 1.0  # No docs needed, so 100% comprehensive

        return len(retrieved_docs & ground_truth_docs) / len(ground_truth_docs)

    def calculate_diversity(
        self,
        retrieved_docs: Set[int],
        ground_truth_docs: Set[int]
    ) -> float:
        """
        Calculate diversity: For multi-passage questions, did we retrieve from multiple sources?

        Diversity score:
        - 1.0 if we retrieved >= number of ground truth docs
        - Proportional if we retrieved fewer
        """
        if len(ground_truth_docs) <= 1:
            # Single document needed, diversity N/A
            return 1.0

        # For multi-passage, we want multiple diverse sources
        num_retrieved = len(retrieved_docs & ground_truth_docs)
        num_needed = len(ground_truth_docs)

        return min(1.0, num_retrieved / num_needed)

    def calculate_logicality(
        self,
        retrieved_docs: Set[int],
        ground_truth_docs: Set[int],
        all_retrieved_docs: Set[int]
    ) -> float:
        """
        Calculate logicality: Ratio of relevant docs to total retrieved docs

        High logicality = most retrieved docs are relevant (low noise)
        """
        if len(all_retrieved_docs) == 0:
            return 0.0

        relevant_count = len(retrieved_docs & ground_truth_docs)
        return relevant_count / len(all_retrieved_docs)

    def calculate_coherence(
        self,
        results: List[Dict],
        ground_truth_docs: Set[int]
    ) -> float:
        """
        Calculate coherence: Are relevant chunks grouped together in results?

        Uses average precision-like metric:
        - Higher score if relevant docs appear earlier in ranking
        - Higher score if relevant docs are clustered together
        """
        if len(ground_truth_docs) == 0:
            return 1.0

        # Extract doc indices in order
        doc_indices_ordered = []
        for result in results:
            doc_id = result.get("id", "").split("_")[0]
            try:
                doc_indices_ordered.append(int(doc_id))
            except (ValueError, AttributeError):
                pass

        if len(doc_indices_ordered) == 0:
            return 0.0

        # Calculate average precision
        relevant_count = 0
        precision_sum = 0.0

        for i, doc_idx in enumerate(doc_indices_ordered):
            if doc_idx in ground_truth_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

        if relevant_count == 0:
            return 0.0

        avg_precision = precision_sum / len(ground_truth_docs)
        return avg_precision

    async def evaluate_question_set(
        self,
        qa_df: pd.DataFrame,
        question_type: str,
        top_k: int = 10
    ) -> Dict:
        """Evaluate on a set of questions"""
        print(f"\n{'='*80}")
        print(f"Evaluating {question_type} Questions")
        print(f"{'='*80}")

        metrics = {
            "relevance": [],
            "comprehensiveness": [],
            "diversity": [],
            "logicality": [],
            "coherence": []
        }

        for idx, row in qa_df.iterrows():
            question = row["question"]

            # Get ground truth docs
            if "document_index" in row:
                if question_type == "multi-passage":
                    # Multi-passage questions may reference multiple docs
                    ground_truth_docs = {int(row["document_index"])}
                else:
                    ground_truth_docs = {int(row["document_index"])}
            else:
                ground_truth_docs = set()  # No-answer questions

            # Retrieve
            results = await self.retrieve_for_query(question, top_k=top_k)

            # Extract retrieved doc indices
            all_retrieved_docs = self.extract_doc_indices_from_results(results)

            # Calculate metrics
            relevance = self.calculate_relevance(all_retrieved_docs, ground_truth_docs)
            comprehensiveness = self.calculate_comprehensiveness(all_retrieved_docs, ground_truth_docs)
            diversity = self.calculate_diversity(all_retrieved_docs, ground_truth_docs)
            logicality = self.calculate_logicality(all_retrieved_docs, ground_truth_docs, all_retrieved_docs)
            coherence = self.calculate_coherence(results, ground_truth_docs)

            metrics["relevance"].append(relevance["f1"])
            metrics["comprehensiveness"].append(comprehensiveness)
            metrics["diversity"].append(diversity)
            metrics["logicality"].append(logicality)
            metrics["coherence"].append(coherence)

            # Print sample
            if idx < 3:
                print(f"\nSample {idx+1}:")
                print(f"  Question: {question[:80]}...")
                print(f"  Ground truth docs: {ground_truth_docs}")
                print(f"  Retrieved docs: {all_retrieved_docs}")
                print(f"  Relevance F1: {relevance['f1']:.3f}")
                print(f"  Comprehensiveness: {comprehensiveness:.3f}")
                print(f"  Diversity: {diversity:.3f}")
                print(f"  Logicality: {logicality:.3f}")
                print(f"  Coherence: {coherence:.3f}")

        # Aggregate metrics
        results = {
            "question_type": question_type,
            "num_questions": len(qa_df),
            "metrics": {
                "relevance": {
                    "mean": float(np.mean(metrics["relevance"])),
                    "std": float(np.std(metrics["relevance"])),
                    "median": float(np.median(metrics["relevance"]))
                },
                "comprehensiveness": {
                    "mean": float(np.mean(metrics["comprehensiveness"])),
                    "std": float(np.std(metrics["comprehensiveness"])),
                    "median": float(np.median(metrics["comprehensiveness"]))
                },
                "diversity": {
                    "mean": float(np.mean(metrics["diversity"])),
                    "std": float(np.std(metrics["diversity"])),
                    "median": float(np.median(metrics["diversity"]))
                },
                "logicality": {
                    "mean": float(np.mean(metrics["logicality"])),
                    "std": float(np.std(metrics["logicality"])),
                    "median": float(np.median(metrics["logicality"]))
                },
                "coherence": {
                    "mean": float(np.mean(metrics["coherence"])),
                    "std": float(np.std(metrics["coherence"])),
                    "median": float(np.median(metrics["coherence"]))
                }
            }
        }

        return results

    async def run_full_evaluation(self, top_k: int = 10):
        """Run complete evaluation"""
        print("\n" + "="*80)
        print("BiG-RAG Single-Topic Evaluation")
        print("="*80)

        # Build graph if needed
        if self.bigrag is None:
            await self.build_graph()

        # Evaluate each question type
        results = {}

        # Single-passage questions
        results["single_passage"] = await self.evaluate_question_set(
            self.single_passage_qa,
            "single-passage",
            top_k=top_k
        )

        # Multi-passage questions
        results["multi_passage"] = await self.evaluate_question_set(
            self.multi_passage_qa,
            "multi-passage",
            top_k=top_k
        )

        # No-answer questions
        results["no_answer"] = await self.evaluate_question_set(
            self.no_answer_qa,
            "no-answer",
            top_k=top_k
        )

        # Save results
        output_file = self.working_dir / "evaluation_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        for question_type, data in results.items():
            print(f"\n{question_type.upper().replace('_', ' ')} ({data['num_questions']} questions):")
            print("-" * 80)

            for metric_name, metric_data in data["metrics"].items():
                print(f"  {metric_name.capitalize():20s}: {metric_data['mean']:.3f} ± {metric_data['std']:.3f} (median: {metric_data['median']:.3f})")

        # Overall averages
        print(f"\n{'='*80}")
        print("OVERALL AVERAGES (across all question types):")
        print("="*80)

        all_metrics = ["relevance", "comprehensiveness", "diversity", "logicality", "coherence"]
        for metric in all_metrics:
            values = [results[qt]["metrics"][metric]["mean"] for qt in results.keys()]
            print(f"  {metric.capitalize():20s}: {np.mean(values):.3f}")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate BiG-RAG on Single-Topic dataset")
    parser.add_argument("--data_source", type=str, default="Single-Topic", help="Dataset name")
    parser.add_argument("--working_dir", type=str, default="../expr", help="Working directory")
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild graph even if exists")

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

    evaluator = SingleTopicEvaluator(
        data_source=args.data_source,
        working_dir=args.working_dir
    )

    # Check if graph exists (check for vdb_entities.json which is created by new implementation)
    graph_exists = (Path(args.working_dir) / args.data_source / "vdb_entities.json").exists()

    if args.rebuild or not graph_exists:
        await evaluator.build_graph()
    else:
        print(f"\n✓ Using existing graph at {Path(args.working_dir) / args.data_source}")
        evaluator.bigrag = BiGRAG(
            working_dir=str(Path(args.working_dir) / args.data_source),
            enable_llm_cache=True
        )

    # Run evaluation
    results = await evaluator.run_full_evaluation(top_k=args.top_k)

    return results


if __name__ == "__main__":
    asyncio.run(main())
