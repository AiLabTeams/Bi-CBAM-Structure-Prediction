import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from sklearn.metrics import r2_score
except ImportError:
    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = PROJECT_ROOT / "benchmark_results_test"
METHOD_DISPLAY = {
    "Bi-CBAM": "Bi-CBAM",
    "Bi-CBAM+Refine": "Bi-CBAM+Refine",
    "BO": "Bayesian Optimization",
    "GA": "Genetic Algorithm",
    "RandomSearch": "Random Search",
    "Trial-and-Error": "Trial-and-Error",
}


def display_method_name(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def parse_json_array_column(series: pd.Series) -> np.ndarray:
    return np.stack(series.apply(lambda x: np.asarray(json.loads(x), dtype=np.float32)).tolist(), axis=0)


def compute_parameter_r2(param_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for method, group in param_df.groupby("method"):
        y_true = parse_json_array_column(group["true_params_norm"])
        y_pred = parse_json_array_column(group["best_params_norm"])
        overall_r2 = float(r2_score(y_true, y_pred))
        per_dim = []
        for dim in range(y_true.shape[1]):
            try:
                per_dim.append(float(r2_score(y_true[:, dim], y_pred[:, dim])))
            except Exception:
                per_dim.append(np.nan)
        rows.append({
            "method": method,
            "param_r2_overall": overall_r2,
            "r2_ia": per_dim[0],
            "r2_phi1": per_dim[1],
            "r2_rad": per_dim[2],
            "r2_times": per_dim[3],
            "r2_phi2": per_dim[4],
            "r2_phi3": per_dim[5],
        })
    return pd.DataFrame(rows)


def build_main_table(summary_df: pd.DataFrame, param_r2_df: pd.DataFrame) -> pd.DataFrame:
    grouped = summary_df.groupby("method", as_index=False).agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
        pcc_mean=("pcc", "mean"),
        pcc_std=("pcc", "std"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        param_mae_mean=("param_mae", "mean"),
        param_mae_std=("param_mae", "std"),
        total_time_sec_mean=("total_time_sec", "mean"),
        total_time_sec_std=("total_time_sec", "std"),
        eval_count_mean=("eval_count", "mean"),
        eval_count_std=("eval_count", "std"),
    )
    table = grouped.merge(param_r2_df, on="method", how="left")
    return table.sort_values(["ssim_mean", "pcc_mean"], ascending=False).reset_index(drop=True)


def build_stratified_tables(summary_df: pd.DataFrame, by: str) -> pd.DataFrame:
    return summary_df.groupby([by, "method"], as_index=False).agg(
        ssim_mean=("ssim", "mean"),
        pcc_mean=("pcc", "mean"),
        objective_mean=("objective", "mean"),
        total_time_sec_mean=("total_time_sec", "mean"),
        eval_count_mean=("eval_count", "mean"),
    )


def build_speedup_table(main_table: pd.DataFrame, baseline_method: str = "Bi-CBAM") -> pd.DataFrame:
    table = main_table.copy()
    if baseline_method not in set(table["method"]):
        return table
    base_time = float(table.loc[table["method"] == baseline_method, "total_time_sec_mean"].iloc[0])
    base_eval = float(table.loc[table["method"] == baseline_method, "eval_count_mean"].iloc[0])
    table["time_speedup_vs_bicbam"] = table["total_time_sec_mean"] / max(base_time, 1e-12)
    table["eval_ratio_vs_bicbam"] = table["eval_count_mean"] / max(base_eval, 1e-12)
    return table


def fmt_pm(mean_val, std_val, digits: int = 4) -> str:
    if pd.isna(mean_val):
        return ""
    if pd.isna(std_val):
        return f"{mean_val:.{digits}f}"
    return f"{mean_val:.{digits}f} ± {std_val:.{digits}f}"


def build_pretty_main_table(main_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in main_table.iterrows():
        rows.append({
            "Method": display_method_name(row["method"]),
            "Objective": fmt_pm(row["objective_mean"], row["objective_std"], 4),
            "SSIM": fmt_pm(row["ssim_mean"], row["ssim_std"], 4),
            "PCC": fmt_pm(row["pcc_mean"], row["pcc_std"], 4),
            "Param MAE": fmt_pm(row["param_mae_mean"], row["param_mae_std"], 4),
            "Param R2": f"{row['param_r2_overall']:.4f}" if not pd.isna(row["param_r2_overall"]) else "",
            "Time (s)": fmt_pm(row["total_time_sec_mean"], row["total_time_sec_std"], 4),
            "Eval Count": fmt_pm(row["eval_count_mean"], row["eval_count_std"], 2),
        })
    return pd.DataFrame(rows)


def save_markdown_table(df: pd.DataFrame, output_path: Path):
    try:
        output_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        output_path.write_text(df.to_string(index=False), encoding="utf-8")


def save_plot(trace_df: pd.DataFrame, output_path: Path, x_col: str, y_col: str, title: str, xlabel: str):
    if plt is None or trace_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in trace_df.groupby("method"):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group[y_col], label=display_method_name(method))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_latex_table(main_table: pd.DataFrame, output_path: Path):
    cols = [
        "method",
        "ssim_mean",
        "pcc_mean",
        "param_r2_overall",
        "param_mae_mean",
        "total_time_sec_mean",
        "eval_count_mean",
    ]
    latex_df = main_table[cols].copy()
    latex_df["method"] = latex_df["method"].map(display_method_name)
    latex_df.columns = ["Method", "SSIM", "PCC", "ParamR2", "ParamMAE", "TimeSec", "EvalCount"]
    output_path.write_text(latex_df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark CSV outputs into paper-ready tables and plots.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--split-name", type=str, default="val")
    args = parser.parse_args()

    subset_tag = f"{args.split_name}_{args.sample_size}"
    summary_path = args.result_dir / f"benchmark_summary_{subset_tag}.csv"
    params_path = args.result_dir / f"benchmark_params_{subset_tag}.csv"
    trace_path = args.result_dir / f"benchmark_trace_aggregate_{subset_tag}.csv"
    if not summary_path.exists() or not params_path.exists():
        raise FileNotFoundError("Missing benchmark output CSV files. Run benchmark_inverse_search.py first.")

    summary_df = pd.read_csv(summary_path)
    param_df = pd.read_csv(params_path)
    trace_df = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()

    param_r2_df = compute_parameter_r2(param_df)
    main_table = build_main_table(summary_df, param_r2_df)
    speedup_table = build_speedup_table(main_table)
    pretty_main_table = build_pretty_main_table(main_table)
    times_table = build_stratified_tables(summary_df, by="times")
    difficulty_table = build_stratified_tables(summary_df, by="difficulty")

    main_table.to_csv(args.result_dir / f"paper_table_main_{subset_tag}.csv", index=False)
    pretty_main_table.to_csv(args.result_dir / f"paper_table_main_pretty_{subset_tag}.csv", index=False)
    speedup_table.to_csv(args.result_dir / f"paper_table_speedup_{subset_tag}.csv", index=False)
    times_table.to_csv(args.result_dir / f"paper_table_by_times_{subset_tag}.csv", index=False)
    difficulty_table.to_csv(args.result_dir / f"paper_table_by_difficulty_{subset_tag}.csv", index=False)
    param_r2_df.to_csv(args.result_dir / f"paper_table_param_r2_{subset_tag}.csv", index=False)
    save_latex_table(main_table, args.result_dir / f"paper_table_main_{subset_tag}.tex")
    save_markdown_table(pretty_main_table, args.result_dir / f"paper_table_main_pretty_{subset_tag}.md")

    if not trace_df.empty:
        trace_df = trace_df.copy()
        trace_df["quality_mean"] = 1.0 - trace_df["best_objective_mean"]
        save_plot(
            trace_df,
            args.result_dir / f"convergence_vs_eval_{subset_tag}.png",
            x_col="step",
            y_col="quality_mean",
            title=f"Convergence vs Evaluations ({args.split_name}, {args.sample_size} samples)",
            xlabel="Forward Evaluations",
        )
        save_plot(
            trace_df,
            args.result_dir / f"convergence_vs_time_{subset_tag}.png",
            x_col="elapsed_sec_mean",
            y_col="quality_mean",
            title=f"Convergence vs Time ({args.split_name}, {args.sample_size} samples)",
            xlabel="Wall-clock Time (s)",
        )


if __name__ == "__main__":
    main()
