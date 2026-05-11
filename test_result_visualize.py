import os
import sys
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.hyperparams import HYPERPARAMS

def parse_metrics_report(file_path):
    metrics = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                key = key.strip()
                val = val.strip()

                # Convert string values to float if possible, else keep string
                try:
                    val = float(val)
                except ValueError:
                    # Ignore non-numeric metrics like "N/A" or "Error" for plotting
                    val = np.nan

                metrics[key] = val
    return metrics


def main():
    # Base paths
    project_root = Path(__file__).resolve().parent
    experiments_dir = project_root / "experiments"

    # 1. Read summary CSV (Optional, but user requested it)
    summary_csv_path = experiments_dir / "summary_major_3.csv"
    if summary_csv_path.exists():
        print(f"[Info] Reading summary CSV from {summary_csv_path}")
        summary_df = pd.read_csv(summary_csv_path)
        print("Summary CSV Content:")
        print(summary_df.head())
    else:
        print(f"[Warning] Summary CSV not found at {summary_csv_path}")

    # 2. Gather metrics from evaluation text files
    search_pattern = str(
        experiments_dir / f"{HYPERPARAMS['major_version']}.*.*" / "evaluation" / "*_metrics_report.txt"
    )
    report_files = glob.glob(search_pattern)

    if not report_files:
        print(f"[Error] No metrics report files found matching {search_pattern}")
        sys.exit(1)

    print(f"\n[Info] Found {len(report_files)} metrics report files.")

    all_metrics = []

    for file_path in report_files:
        # Extract model name from the filename (e.g., MLP_metrics_report.txt -> MLP)
        filename = os.path.basename(file_path)
        model_name = filename.replace("_metrics_report.txt", "")

        metrics_dict = parse_metrics_report(file_path)
        metrics_dict["Model"] = model_name
        all_metrics.append(metrics_dict)

    df_metrics = pd.DataFrame(all_metrics)

    # Drop rows where Model is not recognized or there are empty metrics
    df_metrics = df_metrics.dropna(how="all", axis=1)

    print("\n[Info] Parsed Metrics DataFrame:")
    print(df_metrics)

    # 3. Create Visualization Directory
    save_dir = experiments_dir / f"visualizations_major_{HYPERPARAMS['major_version']}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4. Generate Comparative Plots
    sns.set_theme(style="whitegrid", context="paper")

    # Identify metric columns (excluding "Model")
    metric_cols = [c for c in df_metrics.columns if c != "Model"]

    # Group metrics for better visualization
    # Group 1: Error Metrics (Lower is better)
    error_metrics = [
        c
        for c in metric_cols
        if "MAE" in c or "RMSE" in c or "MAPE" in c or "Max Error" in c
    ]
    # Group 2: R2 and PICP (Higher is better)
    score_metrics = [
        c
        for c in metric_cols
        if "R2 Score" in c or "PICP" in c or "Trend Consistency" in c
    ]
    # Group 3: Computational Metrics
    comp_metrics = [
        c
        for c in metric_cols
        if "Inference Time" in c or "Parameter Count" in c or "FLOPs" in c
    ]

    # Plot Error Metrics
    for metric in error_metrics:
        plt.figure(figsize=(10, 6))

        # Sort values for a neat bar chart
        df_sorted = df_metrics[["Model", metric]].dropna().sort_values(by=metric)

        ax = sns.barplot(
            x="Model",
            y=metric,
            hue="Model",
            data=df_sorted,
            palette="viridis",
            legend=False,
        )
        plt.title(f"Model Comparison: {metric}", fontsize=14, fontweight="bold")
        plt.ylabel(metric)
        plt.xticks(rotation=45)

        # Add values on top of bars
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.4f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="baseline",
                fontsize=10,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

        plt.tight_layout()
        plt.savefig(
            save_dir
            / f"{metric.split(' ')[1].replace('(%)', '').strip()}_comparison.png",
            dpi=300,
        )
        plt.close()

    # Plot Score Metrics
    for metric in score_metrics:
        plt.figure(figsize=(10, 6))

        # Sort values descending (higher is better)
        df_sorted = (
            df_metrics[["Model", metric]]
            .dropna()
            .sort_values(by=metric, ascending=False)
        )

        ax = sns.barplot(
            x="Model",
            y=metric,
            hue="Model",
            data=df_sorted,
            palette="mako",
            legend=False,
        )
        plt.title(f"Model Comparison: {metric}", fontsize=14, fontweight="bold")
        plt.ylabel(metric)
        plt.xticks(rotation=45)

        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.4f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="baseline",
                fontsize=10,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

        plt.tight_layout()
        safe_name = (
            metric.replace("%", "")
            .replace("/", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
            .replace(" ", "_")
        )
        plt.savefig(save_dir / f"{safe_name}_comparison.png", dpi=300)
        plt.close()

    # Plot Computational Metrics
    for metric in comp_metrics:
        plt.figure(figsize=(10, 6))

        df_sorted = df_metrics[["Model", metric]].dropna().sort_values(by=metric)

        if df_sorted.empty:
            plt.close()
            continue

        ax = sns.barplot(
            x="Model",
            y=metric,
            hue="Model",
            data=df_sorted,
            palette="rocket",
            legend=False,
        )
        plt.title(f"Model Comparison: {metric}", fontsize=14, fontweight="bold")
        plt.ylabel(metric)
        plt.xticks(rotation=45)

        # Use log scale if numbers vary widely (e.g. FLOPs or Params)
        if df_sorted[metric].max() > 100 * df_sorted[metric].min():
            plt.yscale("log")
            plt.ylabel(metric + " (Log Scale)")

        for p in ax.patches:
            # Format nicely for large numbers
            val = p.get_height()
            text = f"{val:.0f}" if val > 10 else f"{val:.4f}"
            ax.annotate(
                text,
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="baseline",
                fontsize=10,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

        plt.tight_layout()
        safe_name = (
            metric.replace("%", "")
            .replace("/", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
            .replace(" ", "_")
        )
        plt.savefig(save_dir / f"{safe_name}_comparison.png", dpi=300)
        plt.close()

    # Save the consolidated metrics to a single CSV
    consolidated_csv_path = (
        save_dir / f"consolidated_metrics_major_{HYPERPARAMS['major_version']}.csv"
    )
    df_metrics.to_csv(consolidated_csv_path, index=False)
    print(f"\n[Success] All comparisons plotted and saved to {save_dir}")
    print(f"[Success] Consolidated metrics saved to {consolidated_csv_path}")


if __name__ == "__main__":
    main()
