"""
Result Export Utilities for BiG-RAG Evaluation

Provides functions to export evaluation results in various formats:
- LaTeX tables for research papers
- CSV for Excel/Google Sheets
- JSON for archiving
- Markdown for documentation
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional


# ==============================================================================
# LaTeX Export
# ==============================================================================

def export_to_latex_table(
    results: Dict[str, Dict[str, float]],
    caption: str = "Evaluation Results",
    label: str = "tab:evaluation",
    metrics: Optional[List[str]] = None,
    bold_best: bool = True,
    decimal_places: int = 3
) -> str:
    """
    Export comparison results to LaTeX table

    Args:
        results: Dictionary mapping config_name -> {metric_name: score}
            Example: {
                "BiG-RAG (Hybrid)": {"precision@5": 0.823, "recall@5": 0.901},
                "BiG-RAG (Local)": {"precision@5": 0.756, "recall@5": 0.834}
            }
        caption: Table caption
        label: LaTeX label for referencing
        metrics: List of metrics to include (if None, uses all)
        bold_best: Whether to bold the best score in each column
        decimal_places: Number of decimal places

    Returns:
        LaTeX table string

    Example:
        results = {
            "BiG-RAG (Hybrid)": {"precision@5": 0.823, "recall@5": 0.901, "F1@5": 0.860},
            "BiG-RAG (Local)": {"precision@5": 0.756, "recall@5": 0.834, "F1@5": 0.793},
            "BM25": {"precision@5": 0.612, "recall@5": 0.701, "F1@5": 0.654}
        }
        latex = export_to_latex_table(results, caption="Retrieval Performance")
        print(latex)
    """
    if not results:
        raise ValueError("Results dictionary is empty")

    # Determine metrics to include
    if metrics is None:
        # Get all unique metrics across all configurations
        all_metrics = set()
        for config_results in results.values():
            all_metrics.update(config_results.keys())
        metrics = sorted(list(all_metrics))

    # Determine which configuration has best score for each metric (if bold_best)
    best_scores = {}
    if bold_best:
        for metric in metrics:
            scores = []
            for config_name, config_results in results.items():
                if metric in config_results:
                    scores.append((config_name, config_results[metric]))

            if scores:
                best_config, best_score = max(scores, key=lambda x: x[1])
                best_scores[metric] = (best_config, best_score)

    # Build LaTeX table
    config_names = list(results.keys())
    num_cols = len(metrics) + 1  # +1 for configuration name column

    # Table header
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\small")  # Use small font for better fit
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")

    # Column specification (l for left-aligned config names, r for right-aligned numbers)
    col_spec = "l" + "r" * len(metrics)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    # Header row
    header_row = "Configuration & " + " & ".join([f"\\textbf{{{m}}}" for m in metrics]) + " \\\\"
    latex.append(header_row)
    latex.append("\\midrule")

    # Data rows
    for config_name in config_names:
        config_results = results[config_name]
        row_values = [config_name]

        for metric in metrics:
            if metric in config_results:
                score = config_results[metric]
                score_str = f"{score:.{decimal_places}f}"

                # Bold if best score
                if bold_best and metric in best_scores:
                    best_config, best_score = best_scores[metric]
                    if config_name == best_config:
                        score_str = f"\\textbf{{{score_str}}}"

                row_values.append(score_str)
            else:
                row_values.append("-")  # Missing value

        row_str = " & ".join(row_values) + " \\\\"
        latex.append(row_str)

    # Table footer
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def export_to_latex_comparison_table(
    results: List[Dict[str, Any]],
    caption: str = "Configuration Comparison",
    label: str = "tab:comparison"
) -> str:
    """
    Export detailed comparison table with statistical tests

    Args:
        results: List of comparison results from compare_configurations_statistical
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table with mean, std, p-value, effect size

    Example:
        # After running statistical comparisons
        comparisons = [...]  # List of comparison results
        latex = export_to_latex_comparison_table(comparisons)
    """
    # Implementation left for future enhancement
    raise NotImplementedError("Detailed comparison tables not yet implemented")


# ==============================================================================
# CSV Export
# ==============================================================================

def export_to_csv(
    results: Dict[str, Dict[str, float]],
    output_file: str,
    metrics: Optional[List[str]] = None
) -> str:
    """
    Export results to CSV file

    Args:
        results: Dictionary mapping config_name -> {metric_name: score}
        output_file: Path to output CSV file
        metrics: List of metrics to include (if None, uses all)

    Returns:
        Path to saved CSV file

    Example:
        results = {...}
        export_to_csv(results, "evaluation_results.csv")
    """
    if not results:
        raise ValueError("Results dictionary is empty")

    # Determine metrics
    if metrics is None:
        all_metrics = set()
        for config_results in results.values():
            all_metrics.update(config_results.keys())
        metrics = sorted(list(all_metrics))

    # Prepare output path
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header row
        header = ["Configuration"] + metrics
        writer.writerow(header)

        # Data rows
        for config_name, config_results in results.items():
            row = [config_name]
            for metric in metrics:
                row.append(config_results.get(metric, ""))
            writer.writerow(row)

    return str(output_path)


def export_to_csv_with_stats(
    results: Dict[str, Dict[str, float]],
    output_file: str,
    include_rank: bool = True
) -> str:
    """
    Export results to CSV with additional statistics

    Adds columns for:
    - Rank for each metric
    - Average rank across metrics
    - Overall score (normalized average)

    Args:
        results: Dictionary mapping config_name -> {metric_name: score}
        output_file: Path to output CSV file
        include_rank: Whether to include ranking columns

    Returns:
        Path to saved CSV file
    """
    # Implementation left for future enhancement
    raise NotImplementedError("CSV with statistics not yet implemented")


# ==============================================================================
# Markdown Export
# ==============================================================================

def export_to_markdown_table(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    decimal_places: int = 3,
    bold_best: bool = True
) -> str:
    """
    Export results to Markdown table

    Args:
        results: Dictionary mapping config_name -> {metric_name: score}
        metrics: List of metrics to include
        decimal_places: Number of decimal places
        bold_best: Whether to bold best scores

    Returns:
        Markdown table string

    Example:
        results = {...}
        md = export_to_markdown_table(results)
        print(md)
        # Can be directly used in README.md or documentation
    """
    if not results:
        raise ValueError("Results dictionary is empty")

    # Determine metrics
    if metrics is None:
        all_metrics = set()
        for config_results in results.values():
            all_metrics.update(config_results.keys())
        metrics = sorted(list(all_metrics))

    # Find best scores
    best_scores = {}
    if bold_best:
        for metric in metrics:
            scores = []
            for config_name, config_results in results.items():
                if metric in config_results:
                    scores.append((config_name, config_results[metric]))
            if scores:
                best_config, best_score = max(scores, key=lambda x: x[1])
                best_scores[metric] = (best_config, best_score)

    # Build Markdown table
    lines = []

    # Header
    header = "| Configuration | " + " | ".join(metrics) + " |"
    lines.append(header)

    # Separator
    separator = "|" + "---|" * (len(metrics) + 1)
    lines.append(separator)

    # Data rows
    for config_name, config_results in results.items():
        row_values = [config_name]

        for metric in metrics:
            if metric in config_results:
                score = config_results[metric]
                score_str = f"{score:.{decimal_places}f}"

                # Bold if best
                if bold_best and metric in best_scores:
                    best_config, best_score = best_scores[metric]
                    if config_name == best_config:
                        score_str = f"**{score_str}**"

                row_values.append(score_str)
            else:
                row_values.append("-")

        row_str = "| " + " | ".join(row_values) + " |"
        lines.append(row_str)

    return "\n".join(lines)


# ==============================================================================
# JSON Export (for archiving)
# ==============================================================================

def export_to_json(
    results: Dict[str, Any],
    output_file: str,
    pretty: bool = True
) -> str:
    """
    Export results to JSON file

    Args:
        results: Results dictionary
        output_file: Path to output JSON file
        pretty: Whether to use pretty printing (indented)

    Returns:
        Path to saved JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            json.dump(results, f, ensure_ascii=False)

    return str(output_path)


# ==============================================================================
# Combined Export
# ==============================================================================

def export_all_formats(
    results: Dict[str, Dict[str, float]],
    output_dir: str,
    base_name: str = "evaluation_results",
    caption: str = "Evaluation Results",
    formats: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Export results in multiple formats at once

    Args:
        results: Dictionary mapping config_name -> {metric_name: score}
        output_dir: Output directory
        base_name: Base filename (without extension)
        caption: Caption for LaTeX table
        formats: List of formats to export ("latex", "csv", "markdown", "json")
                 If None, exports all formats.

    Returns:
        Dictionary mapping format -> output_path

    Example:
        results = {...}
        paths = export_all_formats(
            results,
            output_dir="evaluation_results",
            base_name="bigrag_vs_baselines"
        )
        print(f"LaTeX table saved to: {paths['latex']}")
    """
    if formats is None:
        formats = ["latex", "csv", "markdown", "json"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exported = {}

    if "latex" in formats:
        latex_content = export_to_latex_table(results, caption=caption)
        latex_file = output_path / f"{base_name}.tex"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        exported["latex"] = str(latex_file)

    if "csv" in formats:
        csv_file = output_path / f"{base_name}.csv"
        export_to_csv(results, str(csv_file))
        exported["csv"] = str(csv_file)

    if "markdown" in formats:
        md_content = export_to_markdown_table(results)
        md_file = output_path / f"{base_name}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        exported["markdown"] = str(md_file)

    if "json" in formats:
        json_file = output_path / f"{base_name}.json"
        export_to_json(results, str(json_file))
        exported["json"] = str(json_file)

    return exported
