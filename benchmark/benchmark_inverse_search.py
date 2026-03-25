import argparse
import json
import math
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from benchmark_utils import (
    DATA_TRANSFORM,
    DEFAULT_DATA_ROOT,
    DEFAULT_EXCEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    ForwardEvaluator,
    SampleDataset,
    build_samples,
    evaluate_candidate,
    infer_params,
    load_models,
    project_candidate,
    run_bicbam,
    run_bicbam_refine,
    sample_random_candidate,
    SearchResult,
    set_global_seed,
    split_samples,
    stratified_sample,
)

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
except ImportError:
    ConvergenceWarning = None
    GaussianProcessRegressor = None

def run_random_search(rng: np.random.RandomState, evaluator: ForwardEvaluator, target_img: torch.Tensor, true_params: np.ndarray, budget: int) -> SearchResult:
    start = time.perf_counter()
    best_metrics = None
    best_params = None
    trace = []
    for step in range(1, budget + 1):
        metrics = evaluate_candidate(evaluator, target_img, true_params, sample_random_candidate(rng))
        if best_metrics is None or metrics["objective"] < best_metrics["objective"]:
            best_metrics = metrics
            best_params = metrics["pred_params"]
        trace.append({"step": step, "best_objective": best_metrics["objective"], "eval_count": step, "elapsed_sec": time.perf_counter() - start})
    return SearchResult("RandomSearch", best_params, best_metrics, time.perf_counter() - start, budget, trace)


def build_bruteforce_candidates(budget: int) -> List[np.ndarray]:
    budgets = [budget // 3, budget // 3, budget - 2 * (budget // 3)]
    specs = [
        (1, 3, [0, 1, 2]),
        (2, 4, [0, 1, 2, 4]),
        (3, 5, [0, 1, 2, 4, 5]),
    ]
    candidates: List[np.ndarray] = []

    for allocated, (times_int, active_dims, indices) in zip(budgets, specs):
        allocated = max(allocated, 1)
        levels = max(2, int(math.ceil(allocated ** (1.0 / active_dims))))
        axes = [np.linspace(0.0, 1.0, levels, dtype=np.float32) for _ in range(active_dims)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, active_dims)
        if len(mesh) > allocated:
            take_idx = np.linspace(0, len(mesh) - 1, allocated, dtype=int)
            mesh = mesh[take_idx]
        for row in mesh:
            cand = np.zeros(6, dtype=np.float32)
            cand[3] = times_int / 3.0
            for src_idx, dst_idx in enumerate(indices):
                cand[dst_idx] = row[src_idx]
            candidates.append(project_candidate(cand))

    if len(candidates) > budget:
        take_idx = np.linspace(0, len(candidates) - 1, budget, dtype=int)
        candidates = [candidates[i] for i in take_idx]
    while len(candidates) < budget:
        candidates.append(project_candidate(np.array([0.5, 0.5, 0.5, 2 / 3, 0.5, 0.0], dtype=np.float32)))
    return candidates[:budget]


def run_bruteforce_sweep(evaluator: ForwardEvaluator, target_img: torch.Tensor, true_params: np.ndarray, budget: int) -> SearchResult:
    start = time.perf_counter()
    best_metrics = None
    best_params = None
    trace = []
    candidates = build_bruteforce_candidates(budget)

    for step, cand in enumerate(candidates, start=1):
        metrics = evaluate_candidate(evaluator, target_img, true_params, cand)
        if best_metrics is None or metrics["objective"] < best_metrics["objective"]:
            best_metrics = metrics
            best_params = cand.copy()
        trace.append({"step": step, "best_objective": best_metrics["objective"], "eval_count": step, "elapsed_sec": time.perf_counter() - start})

    return SearchResult("Trial-and-Error", best_params, best_metrics, time.perf_counter() - start, len(candidates), trace)


def mutate_population(rng: np.random.RandomState, pop: np.ndarray, sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=pop.shape).astype(np.float32)
    child = pop + noise
    return np.stack([project_candidate(c) for c in child], axis=0)


def run_genetic_algorithm(rng: np.random.RandomState, evaluator: ForwardEvaluator, target_img: torch.Tensor, true_params: np.ndarray, budget: int, population_size: int = 24, elite_size: int = 4, init_params: np.ndarray = None) -> SearchResult:
    start = time.perf_counter()
    pop = np.stack([sample_random_candidate(rng) for _ in range(population_size)], axis=0)
    if init_params is not None:
        pop[0] = project_candidate(init_params)

    scores = []
    best_metrics = None
    best_params = None
    trace = []
    eval_count = 0
    sigma = 0.12

    while eval_count < budget:
        scores.clear()
        for individual in pop:
            if eval_count >= budget:
                break
            metrics = evaluate_candidate(evaluator, target_img, true_params, individual)
            scores.append((metrics["objective"], individual.copy(), metrics))
            eval_count += 1
            if best_metrics is None or metrics["objective"] < best_metrics["objective"]:
                best_metrics = metrics
                best_params = individual.copy()
            trace.append({"step": eval_count, "best_objective": best_metrics["objective"], "eval_count": eval_count, "elapsed_sec": time.perf_counter() - start})
        scores.sort(key=lambda x: x[0])
        elites = np.stack([item[1] for item in scores[:elite_size]], axis=0)
        parent_idx = rng.randint(0, len(scores), size=(population_size - elite_size, 2))
        children = []
        for i, j in parent_idx:
            p1 = scores[min(i, len(scores) - 1)][1]
            p2 = scores[min(j, len(scores) - 1)][1]
            alpha = rng.uniform(0.25, 0.75)
            child = project_candidate(alpha * p1 + (1 - alpha) * p2)
            children.append(child)
        child_arr = np.stack(children, axis=0) if children else np.empty((0, 6), dtype=np.float32)
        child_arr = mutate_population(rng, child_arr, sigma=sigma) if len(child_arr) else child_arr
        pop = np.concatenate([elites, child_arr], axis=0)
        sigma = max(0.03, sigma * 0.97)
    return SearchResult("GA", best_params, best_metrics, time.perf_counter() - start, eval_count, trace)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    improvement = best - mu - xi
    z = improvement / sigma
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1.0 + np.vectorize(lambda x: math.erf(x / np.sqrt(2.0)))(z))
    return improvement * cdf + sigma * pdf


def run_bayesian_optimization(rng: np.random.RandomState, evaluator: ForwardEvaluator, target_img: torch.Tensor, true_params: np.ndarray, budget: int, init_params: np.ndarray = None, n_init: int = 12, candidate_pool: int = 256) -> SearchResult:
    if GaussianProcessRegressor is None:
        return run_random_search(rng, evaluator, target_img, true_params, budget)

    start = time.perf_counter()
    X, y = [], []
    best_metrics = None
    best_params = None
    trace = []

    init_points = [sample_random_candidate(rng) for _ in range(max(n_init - (1 if init_params is not None else 0), 0))]
    if init_params is not None:
        init_points.insert(0, project_candidate(init_params))

    eval_count = 0
    for cand in init_points:
        if eval_count >= budget:
            break
        metrics = evaluate_candidate(evaluator, target_img, true_params, cand)
        X.append(cand)
        y.append(metrics["objective"])
        eval_count += 1
        if best_metrics is None or metrics["objective"] < best_metrics["objective"]:
            best_metrics = metrics
            best_params = cand.copy()
        trace.append({"step": eval_count, "best_objective": best_metrics["objective"], "eval_count": eval_count, "elapsed_sec": time.perf_counter() - start})

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(6), length_scale_bounds=(1e-6, 1e3), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=rng.randint(0, 100000))

    while eval_count < budget:
        with warnings.catch_warnings():
            if ConvergenceWarning is not None:
                warnings.filterwarnings(
                    "ignore",
                    category=ConvergenceWarning,
                    module="sklearn.gaussian_process.kernels",
                )
            gp.fit(np.asarray(X), np.asarray(y))
        pool = np.stack([sample_random_candidate(rng) for _ in range(candidate_pool)], axis=0)
        mu, std = gp.predict(pool, return_std=True)
        ei = expected_improvement(mu, std, min(y))
        cand = pool[int(np.argmax(ei))]
        metrics = evaluate_candidate(evaluator, target_img, true_params, cand)
        X.append(cand)
        y.append(metrics["objective"])
        eval_count += 1
        if metrics["objective"] < best_metrics["objective"]:
            best_metrics = metrics
            best_params = cand.copy()
        trace.append({"step": eval_count, "best_objective": best_metrics["objective"], "eval_count": eval_count, "elapsed_sec": time.perf_counter() - start})

    return SearchResult("BO", best_params, best_metrics, time.perf_counter() - start, eval_count, trace)


def run_single_target(sample: Dict, models, methods: Sequence[str], budget: int, seed: int) -> List[SearchResult]:
    ds = SampleDataset([sample], transform=DATA_TRANSFORM)
    image_tensor, label_tensor = ds[0]
    target_img = image_tensor
    true_params = label_tensor.numpy()
    initial_params = infer_params(models.inverse, image_tensor, models.device)
    results = []

    for offset, method in enumerate(methods):
        rng = np.random.RandomState(seed + offset)
        evaluator = ForwardEvaluator(models.forward, models.device)
        if method == "bicbam":
            result = run_bicbam(initial_params, evaluator, target_img, true_params)
        elif method == "bicbam_refine":
            result = run_bicbam_refine(rng, initial_params, evaluator, target_img, true_params, budget=budget)
        elif method in {"trial", "bruteforce", "sweep"}:
            result = run_bruteforce_sweep(evaluator, target_img, true_params, budget=budget)
        elif method == "random":
            result = run_random_search(rng, evaluator, target_img, true_params, budget=budget)
        elif method == "ga":
            result = run_genetic_algorithm(rng, evaluator, target_img, true_params, budget=budget, init_params=initial_params)
        elif method == "bo":
            result = run_bayesian_optimization(rng, evaluator, target_img, true_params, budget=budget, init_params=initial_params)
        else:
            raise ValueError(f"Unknown method: {method}")
        results.append(result)
    return results


def aggregate_trace(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.groupby(["method", "step"], as_index=False).agg(
        best_objective_mean=("best_objective", "mean"),
        best_objective_std=("best_objective", "std"),
        elapsed_sec_mean=("elapsed_sec", "mean"),
        elapsed_sec_std=("elapsed_sec", "std"),
    )


def summarize_best_attainment(result: SearchResult) -> Dict[str, float]:
    if not result.trace:
        return {
            "best_objective": float(result.best_metrics.get("objective", np.nan)),
            "eval_to_best": float(result.eval_count),
            "time_to_best_sec": float(result.total_time_sec),
        }
    final_best = float(result.trace[-1]["best_objective"])
    best_point = next((point for point in result.trace if float(point["best_objective"]) <= final_best + 1e-12), result.trace[-1])
    return {
        "best_objective": final_best,
        "eval_to_best": float(best_point["eval_count"]),
        "time_to_best_sec": float(best_point["elapsed_sec"]),
    }


def save_selected_images(samples: Sequence[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, sample in enumerate(samples):
        src = Path(sample["image_path"])
        dst = output_dir / f"{idx:03d}_times{sample['times']}_{sample.get('difficulty', 'na')}_{src.name}"
        shutil.copy2(src, dst)
        records.append({
            "sample_index": idx,
            "image_name": sample["image_name"],
            "source_path": str(src),
            "saved_path": str(dst),
            "times": sample["times"],
            "difficulty": sample.get("difficulty", ""),
        })
    pd.DataFrame(records).to_csv(output_dir / "selected_samples.csv", index=False)


def write_csv_variants(df: pd.DataFrame, result_dir: Path, stem: str, subset_tag: str, legacy_tag: str = None):
    df.to_csv(result_dir / f"{stem}_{subset_tag}.csv", index=False)
    if legacy_tag is not None:
        df.to_csv(result_dir / f"{stem}_{legacy_tag}.csv", index=False)


def save_curve_plot(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str, xlabel: str, ylabel: str):
    if plt is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in df.groupby("method"):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group[y_col], label=method)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_curve_plot_variants(df: pd.DataFrame, x_col: str, y_col: str, result_dir: Path, stem: str, subset_tag: str, title: str, xlabel: str, ylabel: str, legacy_tag: str = None):
    save_curve_plot(df, x_col, y_col, result_dir / f"{stem}_{subset_tag}.png", title, xlabel, ylabel)
    if legacy_tag is not None:
        save_curve_plot(df, x_col, y_col, result_dir / f"{stem}_{legacy_tag}.png", title, xlabel, ylabel)


def main():
    parser = argparse.ArgumentParser(description="Benchmark inverse design methods on a stratified test subset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--excel-path", type=Path, default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT.parent / "models")
    parser.add_argument("--model-tag", type=str, default="")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, default="bicbam,bicbam_refine,trial,random,ga,bo")
    parser.add_argument("--result-dir", type=Path, default=PROJECT_ROOT / "benchmark_results")
    parser.add_argument("--split-name", type=str, default="test")
    args = parser.parse_args()

    set_global_seed(args.seed)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    samples = build_samples(args.data_root, args.excel_path)
    splits = split_samples(samples, seed=args.seed)
    if args.split_name not in splits:
        raise ValueError(f"Unknown split_name={args.split_name}. Expected one of: {list(splits.keys())}")
    selected_subset = stratified_sample(splits[args.split_name], n_select=args.sample_size, seed=args.seed)
    models = load_models(model_dir=args.model_dir, model_tag=(args.model_tag if args.model_tag else None))
    subset_tag = f"{args.split_name}_{args.sample_size}"
    legacy_tag = str(args.sample_size) if args.split_name == "test" else None
    selected_dir = args.result_dir / f"selected_images_{subset_tag}"
    save_selected_images(selected_subset, selected_dir)
    if legacy_tag is not None:
        save_selected_images(selected_subset, args.result_dir / f"selected_images_{legacy_tag}")

    summary_rows = []
    trace_rows = []
    sample_rows = []
    total_jobs = len(selected_subset) * args.repeats
    overall_start = time.perf_counter()
    progress = None
    if tqdm is not None:
        progress = tqdm(total=total_jobs, desc=f"Benchmark {args.split_name} {args.sample_size}", unit="case")

    for sample_idx, sample in enumerate(selected_subset):
        for repeat in range(args.repeats):
            results = run_single_target(sample, models, methods, budget=args.budget, seed=args.seed + repeat * 1000 + sample_idx * 17)
            for result in results:
                metrics = {k: v for k, v in result.best_metrics.items() if k not in ("pred_img", "pred_params")}
                best_attainment = summarize_best_attainment(result)
                summary_rows.append({
                    "sample_index": sample_idx,
                    "repeat": repeat,
                    "image_name": sample["image_name"],
                    "split_name": args.split_name,
                    "times": sample["times"],
                    "difficulty": sample.get("difficulty", ""),
                    "method": result.method,
                    "eval_count": result.eval_count,
                    "total_time_sec": result.total_time_sec,
                    "eval_to_best": best_attainment["eval_to_best"],
                    "time_to_best_sec": best_attainment["time_to_best_sec"],
                    **metrics,
                })
                sample_rows.append({
                    "sample_index": sample_idx,
                    "repeat": repeat,
                    "image_name": sample["image_name"],
                    "split_name": args.split_name,
                    "method": result.method,
                    "true_params_norm": json.dumps(sample["label"].tolist()),
                    "best_params_norm": json.dumps(result.best_params.tolist()),
                })
                for point in result.trace:
                    trace_rows.append({
                        "sample_index": sample_idx,
                        "repeat": repeat,
                        "image_name": sample["image_name"],
                        "split_name": args.split_name,
                        "method": result.method,
                        **point,
                    })
            completed_jobs = sample_idx * args.repeats + repeat + 1
            elapsed = time.perf_counter() - overall_start
            eta = (elapsed / completed_jobs) * max(total_jobs - completed_jobs, 0) if completed_jobs else 0.0
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
            elif completed_jobs == total_jobs or completed_jobs % max(1, min(10, total_jobs)) == 0:
                print(
                    f"[{completed_jobs}/{total_jobs}] elapsed={elapsed/60:.1f} min, "
                    f"eta={eta/60:.1f} min, current={sample['image_name']}, repeat={repeat + 1}/{args.repeats}"
                )

    if progress is not None:
        progress.close()

    summary_df = pd.DataFrame(summary_rows)
    sample_df = pd.DataFrame(sample_rows)
    trace_df = pd.DataFrame(trace_rows)
    agg_df = summary_df.groupby("method", as_index=False).agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
        pcc_mean=("pcc", "mean"),
        pcc_std=("pcc", "std"),
        param_mae_mean=("param_mae", "mean"),
        param_mae_std=("param_mae", "std"),
        eval_to_best_mean=("eval_to_best", "mean"),
        eval_to_best_std=("eval_to_best", "std"),
        time_to_best_sec_mean=("time_to_best_sec", "mean"),
        time_to_best_sec_std=("time_to_best_sec", "std"),
        total_time_sec_mean=("total_time_sec", "mean"),
        total_time_sec_std=("total_time_sec", "std"),
        eval_count_mean=("eval_count", "mean"),
        eval_count_std=("eval_count", "std"),
    )
    trace_agg_df = aggregate_trace(trace_rows)
    convergence_curve_df = trace_agg_df.copy()
    time_accuracy_curve_df = trace_agg_df.copy()
    convergence_curve_df["quality_mean"] = 1.0 - convergence_curve_df["best_objective_mean"]
    convergence_curve_df["quality_std"] = convergence_curve_df["best_objective_std"]
    time_accuracy_curve_df["quality_mean"] = 1.0 - time_accuracy_curve_df["best_objective_mean"]
    time_accuracy_curve_df["quality_std"] = time_accuracy_curve_df["best_objective_std"]

    write_csv_variants(summary_df, args.result_dir, "benchmark_summary", subset_tag, legacy_tag)
    write_csv_variants(sample_df, args.result_dir, "benchmark_params", subset_tag, legacy_tag)
    write_csv_variants(trace_df, args.result_dir, "benchmark_trace_raw", subset_tag, legacy_tag)
    write_csv_variants(agg_df, args.result_dir, "benchmark_aggregate", subset_tag, legacy_tag)
    write_csv_variants(trace_agg_df, args.result_dir, "benchmark_trace_aggregate", subset_tag, legacy_tag)
    write_csv_variants(convergence_curve_df, args.result_dir, "convergence_curve_params", subset_tag, legacy_tag)
    write_csv_variants(time_accuracy_curve_df, args.result_dir, "time_accuracy_curve_params", subset_tag, legacy_tag)
    save_curve_plot_variants(
        convergence_curve_df,
        x_col="step",
        y_col="quality_mean",
        result_dir=args.result_dir,
        stem="convergence_curve",
        subset_tag=subset_tag,
        legacy_tag=legacy_tag,
        title=f"Convergence Curve on {args.split_name} subset ({args.sample_size} samples)",
        xlabel="Forward Evaluations",
        ylabel="1 - objective",
    )
    save_curve_plot_variants(
        time_accuracy_curve_df,
        x_col="elapsed_sec_mean",
        y_col="quality_mean",
        result_dir=args.result_dir,
        stem="time_accuracy_curve",
        subset_tag=subset_tag,
        legacy_tag=legacy_tag,
        title=f"Time-Accuracy Curve on {args.split_name} subset ({args.sample_size} samples)",
        xlabel="Wall-clock Time (s)",
        ylabel="1 - objective",
    )
    config_payload = {
        "split_name": args.split_name,
        "sample_size": args.sample_size,
        "budget": args.budget,
        "repeats": args.repeats,
        "methods": methods,
    }
    with open(args.result_dir / f"benchmark_config_{subset_tag}.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    if legacy_tag is not None:
        with open(args.result_dir / f"benchmark_config_{legacy_tag}.json", "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)


if __name__ == "__main__":
    main()

