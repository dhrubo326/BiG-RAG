"""
Statistical Testing Utilities for BiG-RAG Evaluation

Provides statistical significance testing for comparing retrieval/answer configurations:
- Paired t-test
- Wilcoxon signed-rank test (non-parametric)
- Bootstrap confidence intervals
- Effect size calculation (Cohen's d)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats


# ==============================================================================
# Statistical Significance Tests
# ==============================================================================

def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Paired t-test to compare two configurations

    Use this when comparing the same queries/questions evaluated on two different
    configurations (e.g., hybrid vs local mode).

    Args:
        scores_a: Scores from configuration A (one score per query)
        scores_b: Scores from configuration B (one score per query, same order)
        alpha: Significance level (default: 0.05 for 95% confidence)

    Returns:
        Dictionary with test results:
        {
            "test": "paired_t_test",
            "t_statistic": float,
            "p_value": float,
            "significant": bool,
            "mean_diff": float,
            "ci_lower": float,
            "ci_upper": float,
            "interpretation": str
        }

    Example:
        hybrid_scores = [0.8, 0.9, 0.7, 0.85]
        local_scores = [0.6, 0.7, 0.65, 0.75]
        result = paired_t_test(hybrid_scores, local_scores)
        print(f"p-value: {result['p_value']}")
        print(f"Significant: {result['significant']}")
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score lists must have same length. "
            f"Got {len(scores_a)} vs {len(scores_b)}"
        )

    if len(scores_a) < 2:
        raise ValueError("Need at least 2 samples for t-test")

    scores_a_arr = np.array(scores_a)
    scores_b_arr = np.array(scores_b)

    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(scores_a_arr, scores_b_arr)

    # Calculate mean difference
    mean_diff = float(np.mean(scores_a_arr - scores_b_arr))

    # Calculate confidence interval for the difference
    diffs = scores_a_arr - scores_b_arr
    ci = stats.t.interval(
        confidence=1-alpha,
        df=len(diffs)-1,
        loc=np.mean(diffs),
        scale=stats.sem(diffs)
    )

    # Interpretation
    if p_value < alpha:
        if mean_diff > 0:
            interpretation = f"Configuration A is significantly better than B (p={p_value:.4f})"
        else:
            interpretation = f"Configuration B is significantly better than A (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference between configurations (p={p_value:.4f})"

    return {
        "test": "paired_t_test",
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(float(ci[0]), 4),
        "ci_upper": round(float(ci[1]), 4),
        "alpha": alpha,
        "interpretation": interpretation
    }


def wilcoxon_signed_rank(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test)

    Use this when:
    - Scores are not normally distributed
    - Sample size is small
    - Data has outliers

    Args:
        scores_a: Scores from configuration A
        scores_b: Scores from configuration B (same order as A)
        alpha: Significance level

    Returns:
        Dictionary with test results (similar to paired_t_test)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")

    if len(scores_a) < 5:
        raise ValueError("Need at least 5 samples for Wilcoxon test")

    scores_a_arr = np.array(scores_a)
    scores_b_arr = np.array(scores_b)

    # Perform Wilcoxon signed-rank test
    try:
        statistic, p_value = stats.wilcoxon(scores_a_arr, scores_b_arr)
    except ValueError as e:
        # All differences are zero
        return {
            "test": "wilcoxon_signed_rank",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "median_diff": 0.0,
            "alpha": alpha,
            "interpretation": "No difference between configurations (all scores identical)"
        }

    # Calculate median difference
    median_diff = float(np.median(scores_a_arr - scores_b_arr))

    # Interpretation
    if p_value < alpha:
        if median_diff > 0:
            interpretation = f"Configuration A is significantly better than B (p={p_value:.4f})"
        else:
            interpretation = f"Configuration B is significantly better than A (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference between configurations (p={p_value:.4f})"

    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "median_diff": round(median_diff, 4),
        "alpha": alpha,
        "interpretation": interpretation
    }


# ==============================================================================
# Confidence Intervals
# ==============================================================================

def bootstrap_confidence_interval(
    scores: List[float],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    statistic: str = "mean"
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for a statistic

    Args:
        scores: List of scores
        n_iterations: Number of bootstrap samples (default: 1000)
        confidence: Confidence level (default: 0.95 for 95% CI)
        statistic: Which statistic to compute ("mean", "median", "std")

    Returns:
        Dictionary with:
        {
            "statistic": float,  # Point estimate
            "ci_lower": float,   # Lower bound of CI
            "ci_upper": float,   # Upper bound of CI
            "confidence": float  # Confidence level
        }

    Example:
        scores = [0.7, 0.8, 0.75, 0.9, 0.85]
        ci = bootstrap_confidence_interval(scores)
        print(f"Mean: {ci['statistic']:.3f} "
              f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
    """
    if len(scores) < 2:
        raise ValueError("Need at least 2 scores for bootstrap")

    scores_arr = np.array(scores)

    # Select statistic function
    stat_funcs = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std
    }

    if statistic not in stat_funcs:
        raise ValueError(f"Unknown statistic: {statistic}. Use: {list(stat_funcs.keys())}")

    stat_func = stat_funcs[statistic]

    # Bootstrap sampling
    bootstrap_stats = []
    rng = np.random.RandomState(42)  # For reproducibility

    for _ in range(n_iterations):
        # Sample with replacement
        sample = rng.choice(scores_arr, size=len(scores_arr), replace=True)
        bootstrap_stats.append(stat_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate point estimate and confidence interval
    point_estimate = stat_func(scores_arr)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)

    return {
        "statistic": round(float(point_estimate), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "confidence": confidence
    }


# ==============================================================================
# Effect Size
# ==============================================================================

def cohens_d(
    scores_a: List[float],
    scores_b: List[float]
) -> Dict[str, Any]:
    """
    Calculate Cohen's d effect size

    Effect size interpretation:
    - |d| < 0.2: Small effect
    - 0.2 <= |d| < 0.5: Medium effect
    - 0.5 <= |d| < 0.8: Large effect
    - |d| >= 0.8: Very large effect

    Args:
        scores_a: Scores from configuration A
        scores_b: Scores from configuration B

    Returns:
        Dictionary with:
        {
            "cohens_d": float,
            "magnitude": str,  # "small", "medium", "large", "very large"
            "interpretation": str
        }

    Example:
        hybrid = [0.8, 0.85, 0.9, 0.75]
        naive = [0.5, 0.55, 0.6, 0.45]
        effect = cohens_d(hybrid, naive)
        print(f"Effect size: {effect['cohens_d']} ({effect['magnitude']})")
    """
    if len(scores_a) < 2 or len(scores_b) < 2:
        raise ValueError("Need at least 2 scores in each group")

    scores_a_arr = np.array(scores_a)
    scores_b_arr = np.array(scores_b)

    # Calculate pooled standard deviation
    n_a = len(scores_a_arr)
    n_b = len(scores_b_arr)
    var_a = np.var(scores_a_arr, ddof=1)
    var_b = np.var(scores_b_arr, ddof=1)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    # Calculate Cohen's d
    mean_diff = np.mean(scores_a_arr) - np.mean(scores_b_arr)
    d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Interpret magnitude
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    interpretation = (
        f"{'Positive' if d > 0 else 'Negative'} {magnitude} effect "
        f"(d={d:.3f}). "
        f"Configuration {'A' if d > 0 else 'B'} performs "
        f"{abs(d):.3f} standard deviations better."
    )

    return {
        "cohens_d": round(float(d), 4),
        "magnitude": magnitude,
        "interpretation": interpretation
    }


# ==============================================================================
# Comprehensive Comparison
# ==============================================================================

def compare_configurations_statistical(
    scores_a: List[float],
    scores_b: List[float],
    config_a_name: str = "Configuration A",
    config_b_name: str = "Configuration B",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Comprehensive statistical comparison of two configurations

    Runs multiple statistical tests and provides a complete comparison report.

    Args:
        scores_a: Scores from configuration A
        scores_b: Scores from configuration B
        config_a_name: Name for configuration A (for reporting)
        config_b_name: Name for configuration B (for reporting)
        alpha: Significance level

    Returns:
        Dictionary with all statistical test results

    Example:
        hybrid_scores = [0.8, 0.85, 0.9, 0.75, 0.82]
        local_scores = [0.6, 0.65, 0.7, 0.55, 0.62]

        comparison = compare_configurations_statistical(
            hybrid_scores,
            local_scores,
            "Hybrid Mode",
            "Local Mode"
        )

        print(comparison["summary"])
    """
    # Descriptive statistics
    descriptive = {
        config_a_name: {
            "mean": round(float(np.mean(scores_a)), 4),
            "median": round(float(np.median(scores_a)), 4),
            "std": round(float(np.std(scores_a, ddof=1)), 4),
            "min": round(float(np.min(scores_a)), 4),
            "max": round(float(np.max(scores_a)), 4)
        },
        config_b_name: {
            "mean": round(float(np.mean(scores_b)), 4),
            "median": round(float(np.median(scores_b)), 4),
            "std": round(float(np.std(scores_b, ddof=1)), 4),
            "min": round(float(np.min(scores_b)), 4),
            "max": round(float(np.max(scores_b)), 4)
        }
    }

    # Statistical tests
    t_test_result = paired_t_test(scores_a, scores_b, alpha)
    wilcoxon_result = wilcoxon_signed_rank(scores_a, scores_b, alpha) if len(scores_a) >= 5 else None
    effect_size = cohens_d(scores_a, scores_b)

    # Confidence intervals
    ci_a = bootstrap_confidence_interval(scores_a)
    ci_b = bootstrap_confidence_interval(scores_b)

    # Generate summary
    mean_a = descriptive[config_a_name]["mean"]
    mean_b = descriptive[config_b_name]["mean"]
    diff = mean_a - mean_b
    improvement_pct = (diff / mean_b * 100) if mean_b > 0 else 0

    summary = f"""
Statistical Comparison: {config_a_name} vs {config_b_name}

Performance:
  {config_a_name}: {mean_a:.4f} (±{descriptive[config_a_name]['std']:.4f})
  {config_b_name}: {mean_b:.4f} (±{descriptive[config_b_name]['std']:.4f})
  Difference: {diff:+.4f} ({improvement_pct:+.2f}%)

Significance:
  Paired t-test: p={t_test_result['p_value']:.4f} ({'Significant' if t_test_result['significant'] else 'Not significant'})
  Effect size: {effect_size['cohens_d']:.3f} ({effect_size['magnitude']})

Recommendation:
  {t_test_result['interpretation']}
  {effect_size['interpretation']}
""".strip()

    return {
        "descriptive_stats": descriptive,
        "t_test": t_test_result,
        "wilcoxon": wilcoxon_result,
        "effect_size": effect_size,
        "confidence_intervals": {
            config_a_name: ci_a,
            config_b_name: ci_b
        },
        "summary": summary
    }
