"""
Visualization Script for BiG-RAG Evaluation Results

Generates charts and plots from evaluation results:
1. Bar charts comparing metrics across question types
2. Radar charts showing overall performance
3. Comparison charts for different retrieval modes
4. Heatmaps for detailed metric analysis

Requirements:
    pip install matplotlib seaborn pandas

Usage:
    python script_visualize_results.py --results expr/Single-Topic/evaluation_results.json
    python script_visualize_results.py --comparative expr/Single-Topic/comparative_results.json
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict


def plot_metric_comparison(results: Dict, output_dir: Path):
    """Plot bar chart comparing metrics across question types"""
    metrics = ["relevance", "comprehensiveness", "diversity", "logicality", "coherence"]
    question_types = ["single_passage", "multi_passage", "no_answer"]

    # Prepare data
    data = []
    for qt in question_types:
        for metric in metrics:
            data.append({
                "Question Type": qt.replace("_", "-"),
                "Metric": metric.capitalize(),
                "Score": results[qt]["metrics"][metric]["mean"],
                "Std": results[qt]["metrics"][metric]["std"]
            })

    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot grouped bar chart
    x = np.arange(len(metrics))
    width = 0.25

    for i, qt in enumerate(question_types):
        qt_data = df[df["Question Type"] == qt.replace("_", "-")]
        scores = qt_data["Score"].values
        stds = qt_data["Std"].values
        ax.bar(x + i * width, scores, width, label=qt.replace("_", "-").title(),
               yerr=stds, capsize=5, alpha=0.8)

    ax.set_xlabel("Metrics", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("BiG-RAG Performance by Question Type", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend(title="Question Type", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / "metric_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'metric_comparison.png'}")
    plt.close()


def plot_radar_chart(results: Dict, output_dir: Path):
    """Plot radar chart for overall performance"""
    metrics = ["relevance", "comprehensiveness", "diversity", "logicality", "coherence"]
    question_types = ["single_passage", "multi_passage", "no_answer"]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))

    for idx, qt in enumerate(question_types):
        ax = axes[idx]

        # Extract scores
        scores = [results[qt]["metrics"][m]["mean"] for m in metrics]
        scores += scores[:1]  # Complete the circle

        # Angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        # Plot
        ax.plot(angles, scores, 'o-', linewidth=2, label=qt.replace("_", "-").title())
        ax.fill(angles, scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(qt.replace("_", "-").title(), fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "radar_charts.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'radar_charts.png'}")
    plt.close()


def plot_heatmap(results: Dict, output_dir: Path):
    """Plot heatmap of metrics across question types"""
    metrics = ["relevance", "comprehensiveness", "diversity", "logicality", "coherence"]
    question_types = ["single_passage", "multi_passage", "no_answer"]

    # Prepare data matrix
    data_matrix = []
    for qt in question_types:
        row = [results[qt]["metrics"][m]["mean"] for m in metrics]
        data_matrix.append(row)

    df = pd.DataFrame(
        data_matrix,
        index=[qt.replace("_", "-").title() for qt in question_types],
        columns=[m.capitalize() for m in metrics]
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Score'}, ax=ax, linewidths=0.5)
    ax.set_title("BiG-RAG Performance Heatmap", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Metrics", fontsize=12, fontweight='bold')
    ax.set_ylabel("Question Type", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'heatmap.png'}")
    plt.close()


def plot_comparative_modes(results: Dict, output_dir: Path):
    """Plot comparison of different retrieval modes"""
    metrics = ["relevance", "comprehensiveness", "diversity", "logicality", "coherence"]
    modes = ["hybrid", "local", "global", "naive"]

    # Prepare data
    data_matrix = []
    for mode in modes:
        row = [results[mode]["metrics"][m]["mean"] for m in metrics]
        data_matrix.append(row)

    df = pd.DataFrame(
        data_matrix,
        index=[m.capitalize() for m in modes],
        columns=[m.capitalize() for m in metrics]
    )

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(metrics))
    width = 0.2

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

    for i, mode in enumerate(modes):
        scores = df.loc[mode.capitalize()].values
        ax.bar(x + i * width, scores, width, label=mode.capitalize(),
               color=colors[i], alpha=0.8)

    ax.set_xlabel("Metrics", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("BiG-RAG: Retrieval Mode Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend(title="Retrieval Mode", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Add horizontal line at y=0.75 (good performance threshold)
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='Good Performance (0.75)')

    plt.tight_layout()
    plt.savefig(output_dir / "mode_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'mode_comparison.png'}")
    plt.close()


def plot_radar_comparative(results: Dict, output_dir: Path):
    """Plot radar chart comparing retrieval modes"""
    metrics = ["relevance", "comprehensiveness", "diversity", "logicality", "coherence"]
    modes = ["hybrid", "local", "global", "naive"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

    # Angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for idx, mode in enumerate(modes):
        # Extract scores
        scores = [results[mode]["metrics"][m]["mean"] for m in metrics]
        scores += scores[:1]  # Complete the circle

        # Plot
        ax.plot(angles, scores, 'o-', linewidth=2, label=mode.capitalize(),
                color=colors[idx], markersize=8)
        ax.fill(angles, scores, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title("Retrieval Mode Comparison (Radar)", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "radar_comparative.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'radar_comparative.png'}")
    plt.close()


def generate_summary_report(results: Dict, output_path: Path):
    """Generate text summary report"""
    report = []
    report.append("="*80)
    report.append("BiG-RAG EVALUATION SUMMARY REPORT")
    report.append("="*80)
    report.append("")

    if "single_passage" in results:
        # Full evaluation results
        report.append("FULL EVALUATION RESULTS")
        report.append("-"*80)
        report.append("")

        for qt in ["single_passage", "multi_passage", "no_answer"]:
            report.append(f"{qt.replace('_', ' ').upper()} ({results[qt]['num_questions']} questions):")
            report.append("")

            for metric in ["relevance", "comprehensiveness", "diversity", "logicality", "coherence"]:
                mean = results[qt]["metrics"][metric]["mean"]
                std = results[qt]["metrics"][metric]["std"]
                median = results[qt]["metrics"][metric]["median"]

                # Performance label
                if metric == "relevance" and qt == "no_answer":
                    # For no-answer questions, low relevance is good
                    perf = "Excellent" if mean < 0.15 else "Good" if mean < 0.30 else "Fair"
                else:
                    perf = "Excellent" if mean > 0.80 else "Good" if mean > 0.60 else "Fair" if mean > 0.40 else "Poor"

                report.append(f"  {metric.capitalize():<20s}: {mean:.3f} ± {std:.3f} (median: {median:.3f}) [{perf}]")

            report.append("")

    elif "hybrid" in results:
        # Comparative results
        report.append("COMPARATIVE EVALUATION RESULTS")
        report.append("-"*80)
        report.append("")

        metrics = ["relevance", "comprehensiveness", "diversity", "logicality", "coherence"]

        # Table header
        report.append(f"{'Metric':<20} {'Hybrid':>12} {'Local':>12} {'Global':>12} {'Naive':>12} {'Best':>12}")
        report.append("-"*80)

        for metric in metrics:
            values = {mode: results[mode]["metrics"][metric]["mean"] for mode in ["hybrid", "local", "global", "naive"]}
            best_mode = max(values, key=values.get)

            row = f"{metric.capitalize():<20}"
            for mode in ["hybrid", "local", "global", "naive"]:
                marker = "*" if mode == best_mode else " "
                row += f" {values[mode]:>11.3f}{marker}"
            row += f" {best_mode.capitalize():>12}"

            report.append(row)

        report.append("")
        report.append("* = Best performance")
        report.append("")

        # Overall summary
        report.append("OVERALL SUMMARY:")
        report.append("-"*80)

        overall_scores = {
            mode: sum(results[mode]["metrics"][m]["mean"] for m in metrics) / len(metrics)
            for mode in ["hybrid", "local", "global", "naive"]
        }

        best_mode = max(overall_scores, key=overall_scores.get)
        report.append(f"Best retrieval mode: {best_mode.upper()} (avg score: {overall_scores[best_mode]:.3f})")
        report.append("")

        if best_mode == "hybrid":
            improvement = (overall_scores["hybrid"] - max(overall_scores["local"], overall_scores["global"], overall_scores["naive"])) / max(overall_scores["local"], overall_scores["global"], overall_scores["naive"]) * 100
            report.append(f"✓ BiG-RAG (hybrid) outperforms best baseline by {improvement:.1f}%")
        else:
            report.append(f"⚠ Baseline ({best_mode}) outperforms BiG-RAG hybrid mode")

    report.append("")
    report.append("="*80)

    # Save report
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize BiG-RAG evaluation results")
    parser.add_argument("--results", type=str, help="Path to evaluation_results.json")
    parser.add_argument("--comparative", type=str, help="Path to comparative_results.json")
    parser.add_argument("--output_dir", type=str, default="./figures", help="Output directory for figures")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    print(f"\n{'='*80}")
    print("BiG-RAG Result Visualization")
    print(f"{'='*80}\n")

    if args.results:
        # Load full evaluation results
        print(f"Loading full evaluation results from: {args.results}")
        with open(args.results, "r") as f:
            results = json.load(f)

        print("\nGenerating visualizations...")
        plot_metric_comparison(results, output_dir)
        plot_radar_chart(results, output_dir)
        plot_heatmap(results, output_dir)
        generate_summary_report(results, output_dir / "summary_report.txt")

    if args.comparative:
        # Load comparative results
        print(f"\nLoading comparative results from: {args.comparative}")
        with open(args.comparative, "r") as f:
            results = json.load(f)

        print("\nGenerating comparative visualizations...")
        plot_comparative_modes(results, output_dir)
        plot_radar_comparative(results, output_dir)
        generate_summary_report(results, output_dir / "comparative_summary.txt")

    print(f"\n{'='*80}")
    print(f"✓ All visualizations saved to: {output_dir.absolute()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
