"""
Result file helpers (CSV output and cleanup).
"""
from datetime import datetime
from typing import Dict

import glob
import os

import pandas as pd


def save_model_comparison_csv(
    all_metrics: Dict[str, Dict[str, float]],
    ticker: str,
    horizon_label: str,
    horizon_suffix: str = "",
    dataset_label: str = "test",
) -> str:
    """Save model comparison metrics to CSV file."""
    if not all_metrics:
        return ""

    rows = []
    baseline_rmse = None
    if "random_walk" in all_metrics:
        baseline_rmse = all_metrics["random_walk"]["RMSE"]

    for model_name, metrics in all_metrics.items():
        if model_name == "random_walk":
            beats_baseline = "Baseline"
        elif baseline_rmse and metrics["RMSE"] < baseline_rmse:
            beats_baseline = "Yes"
        else:
            beats_baseline = "No"
        rows.append({
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "Beats_Baseline": beats_baseline,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("RMSE")

    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if horizon_suffix:
        filename = f"model_comparison_{dataset_label}_{ticker}_{horizon_suffix}_{timestamp}.csv"
    else:
        filename = f"model_comparison_{dataset_label}_{ticker}_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)
    df.to_csv(filepath, index=False)

    return filepath


def clean_old_results(ticker: str) -> None:
    """Delete all old result files for a given ticker before generating new ones."""
    results_dir = os.path.join(os.getcwd(), "results")
    if not os.path.exists(results_dir):
        return

    patterns = [
        f"forecast_{ticker}_*.png",
        f"volatility_forecast_{ticker}_*.png",
        f"volatility_backtest_{ticker}_*.png",
        f"model_comparison_*_{ticker}_*.csv",
        f"model_comparison_{ticker}_*.csv",
        f"eda_*_{ticker}_*.png",
        f"eda_statistics_{ticker}_*.txt",
    ]

    for pattern in patterns:
        filepath_pattern = os.path.join(results_dir, pattern)
        old_files = glob.glob(filepath_pattern)
        for old_file in old_files:
            try:
                os.remove(old_file)
            except Exception:
                pass
