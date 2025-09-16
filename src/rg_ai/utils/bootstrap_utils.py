"""
Bootstrap Utilities for Uncertainty Estimation

This module provides bootstrap functions to calculate standard errors and confidence intervals
for metrics across different stages of the pipeline:
- Movement classification (on_test_epoch_end)
- Segmentation evaluation (evaluate_segmentation)
- Grading pipeline evaluation

Usage:
    from rg_ai.utils.bootstrap_utils import bootstrap_classification_metrics, bootstrap_segmentation_metrics, bootstrap_regression_metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
from dataclasses import dataclass


@dataclass
class BootstrapResults:
    """Container for bootstrap results"""
    mean: float
    std: float
    stderr: float
    ci_lower: float
    ci_upper: float
    bootstrap_samples: np.ndarray
    
    def __repr__(self):
        return f"Bootstrap(mean={self.mean:.4f}, std={self.std:.4f}, CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}])"


def _bootstrap_resample(data: Union[np.ndarray, List], n_bootstrap: int = 1000, random_state: int = 42) -> List[np.ndarray]:
    """
    Generate bootstrap resamples from data
    
    Args:
        data: Input data to resample
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility
        
    Returns:
        List of bootstrap resamples
    """
    if isinstance(data, list):
        data = np.array(data)
    
    n_samples = len(data)
    if n_samples == 0:
        raise ValueError("Cannot bootstrap empty data")
    
    rng = np.random.RandomState(random_state)
    bootstrap_samples = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_samples.append(indices)
    
    return bootstrap_samples


def _calculate_bootstrap_stats(bootstrap_values: np.ndarray, confidence_level: float = 0.95) -> BootstrapResults:
    """
    Calculate bootstrap statistics from bootstrap values
    
    Args:
        bootstrap_values: Array of bootstrap metric values
        confidence_level: Confidence level for CI calculation
        
    Returns:
        BootstrapResults object with statistics
    """
    mean_val = np.mean(bootstrap_values)
    std_val = np.std(bootstrap_values, ddof=1)
    stderr_val = std_val  # Standard error is std of bootstrap distribution
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    return BootstrapResults(
        mean=mean_val,
        std=std_val,
        stderr=stderr_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        bootstrap_samples=bootstrap_values
    )


def bootstrap_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    average: str = 'macro',
    labels: Optional[List[int]] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    exclude_classes: Optional[List[int]] = None
) -> Dict[str, BootstrapResults]:
    """
    Bootstrap classification metrics for uncertainty estimation
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels
        metrics: List of metrics to calculate ['accuracy', 'precision', 'recall', 'f1']
        average: Averaging strategy for sklearn metrics
        labels: List of labels to include in calculation
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CIs
        random_state: Random seed
        exclude_classes: Classes to exclude from calculation (e.g., "Other" class)
        
    Returns:
        Dict mapping metric names to BootstrapResults
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap empty predictions")
    
    # Filter out excluded classes if specified
    if exclude_classes is not None:
        mask = ~np.isin(y_true, exclude_classes)
        if mask.sum() == 0:
            raise ValueError("No samples remaining after excluding classes")
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
    else:
        y_true_filtered = y_true
        y_pred_filtered = y_pred
    
    # Generate bootstrap indices
    bootstrap_indices = _bootstrap_resample(y_true_filtered, n_bootstrap, random_state)
    
    # Define metric functions
    metric_functions = {
        'accuracy': lambda yt, yp: accuracy_score(yt, yp),
        'precision': lambda yt, yp: precision_score(yt, yp, average=average, labels=labels, zero_division=0.0),
        'recall': lambda yt, yp: recall_score(yt, yp, average=average, labels=labels, zero_division=0.0),
        'f1': lambda yt, yp: f1_score(yt, yp, average=average, labels=labels, zero_division=0.0)
    }
    
    # Calculate bootstrap values for each metric
    results = {}
    
    for metric_name in metrics:
        if metric_name not in metric_functions:
            warnings.warn(f"Unknown metric: {metric_name}, skipping")
            continue
        
        metric_func = metric_functions[metric_name]
        bootstrap_values = []
        
        for indices in bootstrap_indices:
            try:
                y_true_boot = y_true_filtered[indices]
                y_pred_boot = y_pred_filtered[indices]
                metric_value = metric_func(y_true_boot, y_pred_boot)
                bootstrap_values.append(metric_value)
            except Exception as e:
                warnings.warn(f"Error calculating {metric_name} for bootstrap sample: {e}")
                # Use NaN for failed calculations
                bootstrap_values.append(np.nan)
        
        # Remove NaN values
        bootstrap_values = np.array(bootstrap_values)
        bootstrap_values = bootstrap_values[~np.isnan(bootstrap_values)]
        
        if len(bootstrap_values) == 0:
            warnings.warn(f"No valid bootstrap samples for {metric_name}")
            continue
        
        results[metric_name] = _calculate_bootstrap_stats(bootstrap_values, confidence_level)
    
    return results


def bootstrap_segmentation_metrics(
    pred_segments_list: List[List],  # List of SegmentationResult per video
    gt_annotations_list: List[List[str]],  # List of GT annotations per video
    segmentation_metric_func: Callable,  # Function like evaluate_segmentation
    metrics: List[str] = ['mof', 'edit_score', 'f1@25'],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, BootstrapResults]:
    """
    Bootstrap segmentation metrics across videos
    
    Args:
        pred_segments_list: List of predicted segments for each video
        gt_annotations_list: List of ground truth annotations for each video
        segmentation_metric_func: Function to calculate segmentation metrics (e.g., evaluate_segmentation)
        metrics: List of metric names to extract from the function results
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CIs
        random_state: Random seed
        
    Returns:
        Dict mapping metric names to BootstrapResults
    """
    n_videos = len(pred_segments_list)
    if n_videos != len(gt_annotations_list):
        raise ValueError("pred_segments_list and gt_annotations_list must have same length")
    
    if n_videos == 0:
        raise ValueError("Cannot bootstrap empty video list")
    
    # Generate bootstrap indices for videos
    bootstrap_indices = _bootstrap_resample(list(range(n_videos)), n_bootstrap, random_state)
    
    # Calculate bootstrap values for each metric
    results = {}
    
    for metric_name in metrics:
        bootstrap_values = []
        
        for video_indices in bootstrap_indices:
            try:
                # Calculate metric for this bootstrap sample of videos
                video_metrics = []
                
                for video_idx in video_indices:
                    pred_segs = pred_segments_list[video_idx]
                    gt_annots = gt_annotations_list[video_idx]
                    
                    if len(gt_annots) == 0:
                        continue
                    
                    video_result = segmentation_metric_func(pred_segs, gt_annots)
                    
                    if metric_name in video_result:
                        video_metrics.append(video_result[metric_name])
                
                # Average across videos in this bootstrap sample
                if video_metrics:
                    bootstrap_values.append(np.mean(video_metrics))
                else:
                    bootstrap_values.append(np.nan)
                    
            except Exception as e:
                warnings.warn(f"Error calculating {metric_name} for bootstrap sample: {e}")
                bootstrap_values.append(np.nan)
        
        # Remove NaN values
        bootstrap_values = np.array(bootstrap_values)
        bootstrap_values = bootstrap_values[~np.isnan(bootstrap_values)]
        
        if len(bootstrap_values) == 0:
            warnings.warn(f"No valid bootstrap samples for {metric_name}")
            continue
        
        results[metric_name] = _calculate_bootstrap_stats(bootstrap_values, confidence_level)
    
    return results


def bootstrap_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str] = ['mae', 'rmse', 'r2', 'pearson_corr'],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, BootstrapResults]:
    """
    Bootstrap regression metrics for uncertainty estimation (for grading pipeline)
    
    Args:
        y_true: True scores
        y_pred: Predicted scores
        metrics: List of metrics to calculate
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CIs
        random_state: Random seed
        
    Returns:
        Dict mapping metric names to BootstrapResults
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap empty predictions")
    
    # Generate bootstrap indices
    bootstrap_indices = _bootstrap_resample(y_true, n_bootstrap, random_state)
    
    # Define metric functions
    metric_functions = {
        'mae': lambda yt, yp: mean_absolute_error(yt, yp),
        'mse': lambda yt, yp: mean_squared_error(yt, yp),
        'rmse': lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        'r2': lambda yt, yp: r2_score(yt, yp),
        'pearson_corr': lambda yt, yp: pearsonr(yt, yp)[0] if len(yt) > 1 else np.nan,
        'spearman_corr': lambda yt, yp: spearmanr(yt, yp)[0] if len(yt) > 1 else np.nan,
        'median_abs_error': lambda yt, yp: np.median(np.abs(yt - yp)),
        'mean_abs_percent_error': lambda yt, yp: np.mean(np.abs((yt - yp) / np.maximum(yt, 0.1))) * 100
    }
    
    # Calculate bootstrap values for each metric
    results = {}
    
    for metric_name in metrics:
        if metric_name not in metric_functions:
            warnings.warn(f"Unknown metric: {metric_name}, skipping")
            continue
        
        metric_func = metric_functions[metric_name]
        bootstrap_values = []
        
        for indices in bootstrap_indices:
            try:
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]
                metric_value = metric_func(y_true_boot, y_pred_boot)
                bootstrap_values.append(metric_value)
            except Exception as e:
                warnings.warn(f"Error calculating {metric_name} for bootstrap sample: {e}")
                bootstrap_values.append(np.nan)
        
        # Remove NaN values
        bootstrap_values = np.array(bootstrap_values)
        bootstrap_values = bootstrap_values[~np.isnan(bootstrap_values)]
        
        if len(bootstrap_values) == 0:
            warnings.warn(f"No valid bootstrap samples for {metric_name}")
            continue
        
        results[metric_name] = _calculate_bootstrap_stats(bootstrap_values, confidence_level)
    
    return results


def print_bootstrap_results(bootstrap_results: Dict[str, BootstrapResults], title: str = "Bootstrap Results"):
    """
    Pretty print bootstrap results
    
    Args:
        bootstrap_results: Dict of metric name to BootstrapResults
        title: Title for the output
    """
    print(f"\n{'='*60}")
    print(f"{title.upper()}")
    print(f"{'='*60}")
    
    for metric_name, result in bootstrap_results.items():
        print(f"{metric_name.upper()}:")
        print(f"  Mean ± SE: {result.mean:.4f} ± {result.stderr:.4f}")
        print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"  Std Dev: {result.std:.4f}")
        print()


def bootstrap_metric_comparison(
    baseline_bootstrap: Dict[str, BootstrapResults],
    comparison_bootstrap: Dict[str, BootstrapResults],
    metric_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare two sets of bootstrap results to determine significant differences
    
    Args:
        baseline_bootstrap: Bootstrap results for baseline method
        comparison_bootstrap: Bootstrap results for comparison method  
        metric_names: Specific metrics to compare (if None, compares all common metrics)
        
    Returns:
        Dict with comparison statistics for each metric
    """
    if metric_names is None:
        metric_names = list(set(baseline_bootstrap.keys()) & set(comparison_bootstrap.keys()))
    
    comparisons = {}
    
    for metric in metric_names:
        if metric not in baseline_bootstrap or metric not in comparison_bootstrap:
            warnings.warn(f"Metric {metric} not found in both bootstrap results")
            continue
        
        baseline_samples = baseline_bootstrap[metric].bootstrap_samples
        comparison_samples = comparison_bootstrap[metric].bootstrap_samples
        
        # Calculate difference distribution
        difference_samples = comparison_samples - baseline_samples[:len(comparison_samples)]
        
        # Check if improvement is significant (CI doesn't include 0)
        diff_ci_lower = np.percentile(difference_samples, 2.5)
        diff_ci_upper = np.percentile(difference_samples, 97.5)
        is_significant = not (diff_ci_lower <= 0 <= diff_ci_upper)
        
        # Calculate effect size (standardized mean difference)
        pooled_std = np.sqrt((baseline_bootstrap[metric].std**2 + comparison_bootstrap[metric].std**2) / 2)
        effect_size = (comparison_bootstrap[metric].mean - baseline_bootstrap[metric].mean) / pooled_std if pooled_std > 0 else 0
        
        comparisons[metric] = {
            'baseline_mean': baseline_bootstrap[metric].mean,
            'comparison_mean': comparison_bootstrap[metric].mean,
            'difference_mean': np.mean(difference_samples),
            'difference_ci_lower': diff_ci_lower,
            'difference_ci_upper': diff_ci_upper,
            'is_significant': is_significant,
            'effect_size': effect_size,
            'p_value_approx': np.mean(difference_samples <= 0) * 2  # Approximate p-value
        }
    
    return comparisons


# Convenience functions for specific use cases

def bootstrap_classification_results(gts: np.ndarray, preds: np.ndarray, 
                                   label_dict: Dict[str, int],
                                   exclude_other: bool = True,
                                   **kwargs) -> Dict[str, BootstrapResults]:
    """
    Convenience function for movement classification bootstrap
    """
    exclude_classes = [label_dict.get("Other")] if exclude_other and "Other" in label_dict else None
    all_labels = list(range(len(label_dict)))
    
    return bootstrap_classification_metrics(
        y_true=gts,
        y_pred=preds,
        labels=all_labels,
        exclude_classes=exclude_classes,
        **kwargs
    )


def bootstrap_segmentation_evaluation(results: List[Dict], 
                                    method_names: List[str],
                                    evaluation_func: Callable,
                                    **kwargs) -> Dict[str, Dict[str, BootstrapResults]]:
    """
    Convenience function for segmentation pipeline bootstrap
    """
    bootstrap_results = {}
    
    for method in method_names:
        pred_segments_list = []
        gt_annotations_list = []
        
        for result in results:
            gt_annotations = result.get("all_gt_annotations", [])
            
            if method == "unfiltered":
                # Handle unfiltered case - need to convert frame predictions to segments
                from rg_ai.keypoints_pipeline.postprocessing.run_segmentation import UnfilteredSegmenter
                
                frame_predictions = result.get("all_predictions", [])
                frame_confidences = result.get("all_confidences", [])
                
                if frame_predictions and gt_annotations:
                    segmenter = UnfilteredSegmenter(min_segment_length=1)
                    segments = segmenter._predictions_to_segments(frame_predictions, frame_confidences)
                    pred_segments_list.append(segments)
                    gt_annotations_list.append(gt_annotations)
            else:
                # Handle regular segmentation methods
                segments = result.get(f"{method}_segments", [])
                
                if segments and gt_annotations:
                    pred_segments_list.append(segments)
                    gt_annotations_list.append(gt_annotations)
        
        if pred_segments_list:
            bootstrap_results[method] = bootstrap_segmentation_metrics(
                pred_segments_list=pred_segments_list,
                gt_annotations_list=gt_annotations_list,
                segmentation_metric_func=evaluation_func,
                **kwargs
            )
    
    return bootstrap_results 