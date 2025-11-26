#!/usr/bin/env python3
"""
Compare performance of all trained models.
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_dir):
    """Load results from all model directories."""
    results = {}

    for model_dir in Path(results_dir).iterdir():
        if model_dir.is_dir():
            results_file = model_dir / 'results.yaml'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    model_results = yaml.safe_load(f)
                    results[model_dir.name] = model_results

    return results


def create_comparison_table(results):
    """Create comparison table of model performances."""
    data = []

    for model_name, model_results in results.items():
        if 'test_metrics' in model_results:
            metrics = model_results['test_metrics']
            data.append({
                'Model': model_name.upper(),
                'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                'F1 Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })

    df = pd.DataFrame(data)
    return df


def plot_comparison(results, save_path=None):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['accuracy', 'f1', 'roc_auc']
    titles = ['Accuracy (%)', 'F1 Score', 'ROC-AUC']

    for ax, metric, title in zip(axes, metrics, titles):
        values = []
        labels = []

        for model_name, model_results in results.items():
            if 'test_metrics' in model_results:
                value = model_results['test_metrics'][metric]
                if metric == 'accuracy':
                    value = value  # Already in percentage
                values.append(value)
                labels.append(model_name.upper())

        bars = ax.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')
        ax.set_ylim(0, 100 if metric == 'accuracy' else 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}' if metric == 'accuracy' else f'{value:.3f}',
                   ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare model performances')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing model results')
    parser.add_argument('--output-dir', type=str, default='./results/comparison',
                       help='Output directory for comparison')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    results = load_results(args.results_dir)

    if not results:
        print("No results found!")
        return

    print(f"Found results for {len(results)} models: {list(results.keys())}")

    # Create comparison table
    comparison_df = create_comparison_table(results)
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 50)
    print(comparison_df.to_string(index=False))

    # Save comparison table
    comparison_df.to_csv(os.path.join(args.output_dir, 'comparison_table.csv'), index=False)

    # Create comparison plots
    plot_path = os.path.join(args.output_dir, 'comparison_plot.png')
    plot_comparison(results, plot_path)

    # Find best model
    best_model = None
    best_f1 = 0
    for model_name, model_results in results.items():
        if 'test_metrics' in model_results:
            f1 = model_results['test_metrics']['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name

    print("\n" + "=" * 50)
    print(f"Best Model: {best_model.upper()} (F1 Score: {best_f1:.4f})")
    print("=" * 50)


if __name__ == '__main__':
    main()
