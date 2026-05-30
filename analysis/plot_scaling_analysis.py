#!/usr/bin/env python3
"""
Scaling Analysis Plot for Qwen 2.5 Instruct Models

分析不同规模的Qwen 2.5 Instruct模型在:
1. 跨语言分类性能 (训练: en/zh/ko, 测试: sw)
2. 多语言通用性能 (MMLU on sw)

绘制两者之间的关系图

Usage:
    python plot_scaling_analysis.py \
        --results_dir ./output_qwen_scaling/scaling_results \
        --test_language sw \
        --output_dir ./output_qwen_scaling/scaling_results
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
import numpy as np

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting will be skipped.")


# Model size order and parameter counts (in billions)
MODEL_SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
PARAM_COUNTS = {
    "0.5B": 0.5,
    "1.5B": 1.5,
    "3B": 3.0,
    "7B": 7.0,
    "14B": 14.0,
    "32B": 32.0
}


def load_classification_results(results_dir: str) -> Dict[str, Dict]:
    """Load classification evaluation results for all model sizes."""
    results = {}
    
    for size in MODEL_SIZES:
        filepath = os.path.join(results_dir, f"classification_{size}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            results[size] = data
            print(f"Loaded classification results for {size}")
        else:
            print(f"Warning: Classification results not found for {size}")
    
    return results


def load_mmlu_results(results_dir: str) -> Dict[str, Dict]:
    """Load MMLU evaluation results for all model sizes."""
    results = {}
    
    for size in MODEL_SIZES:
        filepath = os.path.join(results_dir, f"mmlu_{size}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            results[size] = data
            print(f"Loaded MMLU results for {size}")
        else:
            print(f"Warning: MMLU results not found for {size}")
    
    return results


def extract_metrics(
    classification_results: Dict[str, Dict],
    mmlu_results: Dict[str, Dict],
    test_language: str
) -> Tuple[List[str], List[float], List[float], List[float], List[float], List[float]]:
    """Extract key metrics from results.
    
    Returns:
        sizes: List of model sizes
        params: List of parameter counts
        classification_acc: Classification accuracy on test language
        classification_f1: Classification F1 score on test language
        classification_auc: Classification AUC-ROC on test language
        mmlu_acc: MMLU accuracy on test language
    """
    sizes = []
    params = []
    classification_acc = []
    classification_f1 = []
    classification_auc = []
    mmlu_acc = []
    
    for size in MODEL_SIZES:
        has_classification = size in classification_results
        has_mmlu = size in mmlu_results
        
        if not has_classification and not has_mmlu:
            continue
        
        sizes.append(size)
        params.append(PARAM_COUNTS[size])
        
        # Classification metrics
        if has_classification:
            clf_data = classification_results[size]
            
            # Try different ways to get metrics for test language
            if 'classification' in clf_data:
                # New format with classification key
                clf_metrics = clf_data['classification']
                by_lang = clf_data.get('classification_by_language', {})
            else:
                # Old format
                clf_metrics = clf_data
                by_lang = clf_data.get('by_language', {})
            
            # Get metrics for specific language if available
            if test_language in by_lang:
                lang_metrics = by_lang[test_language]
                classification_acc.append(lang_metrics.get('accuracy', 0.0))
                classification_f1.append(lang_metrics.get('f1', 0.0))
                classification_auc.append(lang_metrics.get('auc_roc', 0.0))
            else:
                # Use overall metrics
                classification_acc.append(clf_metrics.get('accuracy', 0.0))
                classification_f1.append(clf_metrics.get('f1', 0.0))
                classification_auc.append(clf_metrics.get('auc_roc', 0.0))
        else:
            classification_acc.append(None)
            classification_f1.append(None)
            classification_auc.append(None)
        
        # MMLU metrics
        if has_mmlu:
            mmlu_data = mmlu_results[size]
            mmlu_acc.append(mmlu_data.get('accuracy', 0.0))
        else:
            mmlu_acc.append(None)
    
    return sizes, params, classification_acc, classification_f1, classification_auc, mmlu_acc


def plot_scaling_analysis(
    sizes: List[str],
    params: List[float],
    classification_acc: List[float],
    classification_f1: List[float],
    classification_auc: List[float],
    mmlu_acc: List[float],
    test_language: str,
    output_dir: str
):
    """Generate scaling analysis plots."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping plot generation.")
        return
    
    # Filter out None values for plotting
    valid_clf_indices = [i for i, v in enumerate(classification_acc) if v is not None]
    valid_mmlu_indices = [i for i, v in enumerate(mmlu_acc) if v is not None]
    
    # 使用专业的颜色方案
    colors = {
        'classification': '#2563EB',  # Blue
        'f1': '#059669',              # Green  
        'auc': '#7C3AED',             # Purple
        'mmlu': '#DC2626',            # Red
        'correlation': '#F59E0B'       # Orange
    }
    
    # 设置字体
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    
    # ============================================
    # Figure 1: Main Scaling Analysis (2x2 grid)
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Qwen 2.5 Instruct Scaling Analysis\nCross-lingual Classification (train: en/zh/ko → test: {test_language}) vs MMLU ({test_language})', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    x_positions = np.arange(len(sizes))
    
    # ---- Plot 1: Classification Metrics by Model Size ----
    ax = axes[0, 0]
    width = 0.25
    
    if valid_clf_indices:
        valid_sizes = [sizes[i] for i in valid_clf_indices]
        valid_acc = [classification_acc[i] for i in valid_clf_indices]
        valid_f1 = [classification_f1[i] for i in valid_clf_indices]
        valid_auc = [classification_auc[i] for i in valid_clf_indices]
        x = np.arange(len(valid_sizes))
        
        bars1 = ax.bar(x - width, valid_acc, width, label='Accuracy', color=colors['classification'], alpha=0.85)
        bars2 = ax.bar(x, valid_f1, width, label='F1 Score', color=colors['f1'], alpha=0.85)
        bars3 = ax.bar(x + width, valid_auc, width, label='AUC-ROC', color=colors['auc'], alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels(valid_sizes)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Score')
    ax.set_title(f'Classification Performance on {test_language.upper()}', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # ---- Plot 2: MMLU Accuracy by Model Size ----
    ax = axes[0, 1]
    
    if valid_mmlu_indices:
        valid_sizes = [sizes[i] for i in valid_mmlu_indices]
        valid_mmlu = [mmlu_acc[i] for i in valid_mmlu_indices]
        x = np.arange(len(valid_sizes))
        
        bars = ax.bar(x, valid_mmlu, width=0.5, color=colors['mmlu'], alpha=0.85, edgecolor='darkred', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(valid_sizes)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'MMLU Accuracy on {test_language.upper()}', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # ---- Plot 3: Line Plot - Classification vs Model Scale (log scale) ----
    ax = axes[1, 0]
    
    if valid_clf_indices:
        valid_params = [params[i] for i in valid_clf_indices]
        valid_acc = [classification_acc[i] for i in valid_clf_indices]
        valid_f1 = [classification_f1[i] for i in valid_clf_indices]
        valid_auc = [classification_auc[i] for i in valid_clf_indices]
        
        ax.plot(valid_params, valid_acc, 'o-', color=colors['classification'], 
                linewidth=2.5, markersize=10, label='Accuracy', markeredgecolor='white', markeredgewidth=2)
        ax.plot(valid_params, valid_f1, 's--', color=colors['f1'], 
                linewidth=2.5, markersize=10, label='F1 Score', markeredgecolor='white', markeredgewidth=2)
        ax.plot(valid_params, valid_auc, '^:', color=colors['auc'], 
                linewidth=2.5, markersize=10, label='AUC-ROC', markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('Model Parameters (Billions)')
    ax.set_ylabel('Score')
    ax.set_title(f'Classification Scaling Trend on {test_language.upper()}', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)
    
    # ---- Plot 4: Scatter Plot - Classification F1 vs MMLU Accuracy ----
    ax = axes[1, 1]
    
    # Find common valid indices
    common_indices = [i for i in range(len(sizes)) 
                     if classification_f1[i] is not None and mmlu_acc[i] is not None]
    
    if common_indices:
        x_vals = [mmlu_acc[i] for i in common_indices]
        y_vals = [classification_f1[i] for i in common_indices]
        labels = [sizes[i] for i in common_indices]
        param_sizes = [params[i] * 15 + 50 for i in common_indices]  # Size based on params
        
        scatter = ax.scatter(x_vals, y_vals, s=param_sizes, c=colors['correlation'], 
                            alpha=0.7, edgecolors='black', linewidths=1.5)
        
        # Add labels
        for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
            ax.annotate(label, (x, y), xytext=(8, 8), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add trend line if enough points
        if len(common_indices) >= 3:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_vals) - 0.05, max(x_vals) + 0.05, 100)
            ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, linewidth=2, label='Trend')
            
            # Calculate correlation
            correlation = np.corrcoef(x_vals, y_vals)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(f'MMLU Accuracy ({test_language.upper()})')
    ax.set_ylabel(f'Classification F1 ({test_language.upper()})')
    ax.set_title('Relationship: General Capability vs Classification', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'scaling_analysis.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Main plot saved to {os.path.join(output_dir, 'scaling_analysis.png')}")
    
    # ============================================
    # Figure 2: Dual-Axis Line Plot
    # ============================================
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Find common valid indices
    common_indices = [i for i in range(len(sizes)) 
                     if classification_f1[i] is not None and mmlu_acc[i] is not None]
    
    if common_indices:
        valid_sizes = [sizes[i] for i in common_indices]
        valid_params = [params[i] for i in common_indices]
        valid_f1 = [classification_f1[i] for i in common_indices]
        valid_mmlu = [mmlu_acc[i] for i in common_indices]
        x = np.arange(len(valid_sizes))
        
        # Left y-axis: Classification F1
        color1 = colors['f1']
        ax1.set_xlabel('Model Size', fontsize=14)
        ax1.set_ylabel('Classification F1 Score', color=color1, fontsize=14)
        line1 = ax1.plot(x, valid_f1, 'o-', color=color1, linewidth=3, markersize=12, 
                        label='Classification F1', markeredgecolor='white', markeredgewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 1.05)
        
        # Right y-axis: MMLU Accuracy  
        ax2 = ax1.twinx()
        color2 = colors['mmlu']
        ax2.set_ylabel('MMLU Accuracy', color=color2, fontsize=14)
        line2 = ax2.plot(x, valid_mmlu, 's--', color=color2, linewidth=3, markersize=12,
                        label='MMLU Accuracy', markeredgecolor='white', markeredgewidth=2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 1.0)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_sizes)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower right', fontsize=12)
        
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
    
    plt.title(f'Classification vs MMLU Performance Scaling ({test_language.upper()})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_dual_axis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dual-axis plot saved to {os.path.join(output_dir, 'scaling_dual_axis.png')}")


def generate_summary_report(
    sizes: List[str],
    params: List[float],
    classification_acc: List[float],
    classification_f1: List[float],
    classification_auc: List[float],
    mmlu_acc: List[float],
    test_language: str,
    output_dir: str
):
    """Generate a summary JSON report."""
    
    summary = {
        "experiment": "Qwen 2.5 Instruct Scaling Analysis",
        "test_language": test_language,
        "train_languages": ["en", "zh", "ko"],
        "models": []
    }
    
    for i, size in enumerate(sizes):
        model_data = {
            "size": size,
            "parameters_billions": params[i],
            "classification": {
                "accuracy": classification_acc[i],
                "f1_score": classification_f1[i],
                "auc_roc": classification_auc[i]
            },
            "mmlu_accuracy": mmlu_acc[i]
        }
        summary["models"].append(model_data)
    
    # Calculate correlation if enough data points
    valid_indices = [i for i in range(len(sizes)) 
                    if classification_f1[i] is not None and mmlu_acc[i] is not None]
    
    if len(valid_indices) >= 3:
        f1_vals = [classification_f1[i] for i in valid_indices]
        mmlu_vals = [mmlu_acc[i] for i in valid_indices]
        correlation = np.corrcoef(f1_vals, mmlu_vals)[0, 1]
        summary["correlation_f1_mmlu"] = float(correlation)
    
    # Save summary
    output_path = os.path.join(output_dir, "scaling_summary.json")
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTest Language: {test_language.upper()}")
    print(f"Train Languages: en, zh, ko")
    print("\n" + "-" * 80)
    print(f"{'Model':<10} {'Params':<10} {'Clf Acc':<10} {'Clf F1':<10} {'Clf AUC':<10} {'MMLU Acc':<10}")
    print("-" * 80)
    
    for i, size in enumerate(sizes):
        clf_acc_str = f"{classification_acc[i]:.4f}" if classification_acc[i] is not None else "-"
        clf_f1_str = f"{classification_f1[i]:.4f}" if classification_f1[i] is not None else "-"
        clf_auc_str = f"{classification_auc[i]:.4f}" if classification_auc[i] is not None else "-"
        mmlu_str = f"{mmlu_acc[i]:.4f}" if mmlu_acc[i] is not None else "-"
        
        print(f"{size:<10} {params[i]:<10.1f} {clf_acc_str:<10} {clf_f1_str:<10} {clf_auc_str:<10} {mmlu_str:<10}")
    
    print("-" * 80)
    
    if "correlation_f1_mmlu" in summary:
        print(f"\nCorrelation (Classification F1 vs MMLU): {summary['correlation_f1_mmlu']:.4f}")
    
    print("=" * 80)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Qwen 2.5 Scaling Analysis Plot')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing classification_*.json and mmlu_*.json files')
    parser.add_argument('--test_language', type=str, default='sw',
                       help='Test language code (default: sw)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as results_dir)')
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading results from: {args.results_dir}")
    print(f"Test language: {args.test_language}")
    
    # Load results
    classification_results = load_classification_results(args.results_dir)
    mmlu_results = load_mmlu_results(args.results_dir)
    
    if not classification_results and not mmlu_results:
        print("Error: No results found!")
        return
    
    # Extract metrics
    sizes, params, clf_acc, clf_f1, clf_auc, mmlu_acc = extract_metrics(
        classification_results, mmlu_results, args.test_language
    )
    
    if not sizes:
        print("Error: No valid results to plot!")
        return
    
    # Generate summary report
    summary = generate_summary_report(
        sizes, params, clf_acc, clf_f1, clf_auc, mmlu_acc,
        args.test_language, output_dir
    )
    
    # Generate plots
    if HAS_MATPLOTLIB:
        plot_scaling_analysis(
            sizes, params, clf_acc, clf_f1, clf_auc, mmlu_acc,
            args.test_language, output_dir
        )
    else:
        print("Skipping plot generation (matplotlib not available)")
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

