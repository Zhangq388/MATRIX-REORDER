#!/usr/bin/env python3
"""
Evaluate exact and sampled reuse-distance (RD) features for two SpMV backends:
MERBIT and CSR.

Expected schema for every input CSV:

    graph_id,
    exact_r1,...,exact_r10,exact_time_ms,
    sampled_r1,...,sampled_r10,sampled_time_ms,
    speedup

Default data layout:

    data/train_merbit.csv
    data/test_sparse_merbit.csv
    data/test_rmat_merbit.csv
    data/train_csr.csv
    data/test_sparse_csr.csv
    data/test_rmat_csr.csv

For each backend, the script trains and evaluates two independent Ridge models:

    exact RD features   -> log(speedup)
    sampled RD features -> log(speedup)

The same graph folds are shared by both feature variants and both backends, so
all exact-vs-sampled and MERBIT-vs-CSR comparisons are paired and leakage-free.
The sampled-trained model is the intended deployment path; exact RD is retained
as a scientific baseline.

Main experiments per backend:
    A. Grouped cross-validation on augmented SuiteSparse samples.
    B. Train on all augmented samples and test original SuiteSparse graphs from
       the same graph set (auxiliary same-graph result).
    C. Leakage-free held-out SuiteSparse graph evaluation.
    D. Train on all augmented samples and test external RMAT graphs.

Additional outputs include feature-approximation statistics, extraction-time
statistics, gate metrics, cross-feature diagnostics, backend-sensitivity
comparisons, and deployable CSV/JSON/C++ model parameters for both backends.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# Configuration
# =========================================================

FEATURE_COUNT = 10
FEATURE_VARIANTS: Tuple[str, ...] = ("exact", "sampled")
SUPPORTED_KERNELS: Tuple[str, ...] = ("merbit", "csr")
DEFAULT_DATA_FILES: Dict[str, Dict[str, str]] = {
    "merbit": {
        "train": "train_merbit.csv",
        "sparse": "test_sparse_merbit.csv",
        "rmat": "test_rmat_merbit.csv",
    },
    "csr": {
        "train": "train_csr.csv",
        "sparse": "test_sparse_csr.csv",
        "rmat": "test_rmat_csr.csv",
    },
}
RIDGE_ALPHA = 1.0
CV_FOLDS = 5
EPS = 1e-12
ROW_SUM_WARN_TOL = 1e-3
RANDOM_SEED = 20260616

BAND_GATE_PAIRS: List[Tuple[float, float]] = [
    (0.99, 1.01),
    (0.98, 1.02),
    (0.97, 1.03),
    (0.96, 1.04),
    (0.95, 1.05),
    (0.94, 1.06),
    (0.93, 1.07),
]


# =========================================================
# Data structures
# =========================================================


@dataclass(frozen=True)
class GraphFold:
    fold: int
    train_graphs: Tuple[str, ...]
    test_graphs: Tuple[str, ...]


@dataclass(frozen=True)
class KernelBundle:
    kernel: str
    train: pd.DataFrame
    sparse: pd.DataFrame
    rmat: pd.DataFrame


@dataclass
class VariantResult:
    kernel: str
    variant: str
    model_all_aug: Pipeline
    feature_cols: List[str]
    master_summary: pd.DataFrame
    predictions: Dict[str, pd.DataFrame]
    gate_tables: Dict[str, pd.DataFrame]
    coefficient_table: pd.DataFrame


@dataclass
class KernelResult:
    kernel: str
    variants: Dict[str, VariantResult]
    metric_comparison: pd.DataFrame
    cross_feature_summary: pd.DataFrame


# =========================================================
# General utilities
# =========================================================


def natural_graph_key(value: object) -> Tuple[object, ...]:
    """Sort numeric graph IDs numerically and all other IDs lexically."""
    text = str(value).strip()
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def spearman_corr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    finite = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[finite]
    y_pred_arr = y_pred_arr[finite]

    if len(y_true_arr) < 2:
        return np.nan

    y_true_rank = pd.Series(y_true_arr).rank(method="average").to_numpy()
    y_pred_rank = pd.Series(y_pred_arr).rank(method="average").to_numpy()

    a = y_true_rank - y_true_rank.mean()
    b = y_pred_rank - y_pred_rank.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denominator < 1e-15:
        return np.nan
    return float(np.sum(a * b) / denominator)


def pearson_corr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if len(a) < 2 or np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(true_arr), EPS)
    return float(np.mean(np.abs(pred_arr - true_arr) / denominator))


def symmetric_mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(true_arr) + np.abs(pred_arr), EPS)
    return float(np.mean(2.0 * np.abs(pred_arr - true_arr) / denominator))


def _is_number_like(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_int_like(text: str) -> bool:
    return text.isdigit() or (text.startswith("-") and text[1:].isdigit())


def extract_base_graph_id_from_swap(graph_id: object) -> str:
    """
    Remove the final local-swap suffix while preserving underscores in graph names.

    Examples:
        1_0.1_64_256           -> 1
        belgium_osm_0.1_64_256 -> belgium_osm
    """
    text = str(graph_id).strip()
    parts = text.split("_")

    if (
        len(parts) >= 4
        and _is_number_like(parts[-3])
        and _is_int_like(parts[-2])
        and _is_int_like(parts[-1])
    ):
        return "_".join(parts[:-3])

    if len(parts) > 1 and _is_number_like(parts[1]):
        return parts[0]

    return text


def canonical_r_cols() -> List[str]:
    return [f"r{i}" for i in range(1, FEATURE_COUNT + 1)]


def source_r_cols(variant: str) -> List[str]:
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    return [f"{variant}_r{i}" for i in range(1, FEATURE_COUNT + 1)]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# =========================================================
# Loading and validation
# =========================================================


def required_dual_columns() -> List[str]:
    return [
        "graph_id",
        *source_r_cols("exact"),
        "exact_time_ms",
        *source_r_cols("sampled"),
        "sampled_time_ms",
        "speedup",
    ]


def load_dual_feature_dataset(
    path: Path,
    dataset_name: str,
    graph_mode: str,
    kernel: str,
) -> pd.DataFrame:
    """
    graph_mode:
        swap   - augmented graph ID with local-swap suffix;
        sparse - original SuiteSparse ID;
        rmat   - external RMAT ID (prefixed internally to avoid ID collisions).
    """
    kernel = kernel.strip().lower()
    if kernel not in SUPPORTED_KERNELS:
        raise ValueError(f"Unsupported kernel: {kernel}")
    if not path.exists():
        raise FileNotFoundError(f"Missing {kernel}/{dataset_name} file: {path}")

    df = pd.read_csv(path)
    df.columns = [str(column).strip().lower() for column in df.columns]

    missing = [column for column in required_dual_columns() if column not in df.columns]
    if missing:
        raise ValueError(
            f"[{dataset_name}] missing columns: {missing}. "
            f"Actual columns: {list(df.columns)}"
        )

    df = df[required_dual_columns()].copy()
    df["graph_id"] = df["graph_id"].astype(str).str.strip()

    numeric_columns = [
        *source_r_cols("exact"),
        "exact_time_ms",
        *source_r_cols("sampled"),
        "sampled_time_ms",
        "speedup",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_mask = df[numeric_columns].isna().any(axis=1)
    if invalid_mask.any():
        bad_ids = df.loc[invalid_mask, "graph_id"].head(10).tolist()
        raise ValueError(
            f"[{dataset_name}] found {int(invalid_mask.sum())} rows with missing or "
            f"non-numeric values. Example graph IDs: {bad_ids}"
        )

    if df["graph_id"].duplicated().any():
        duplicates = df.loc[df["graph_id"].duplicated(keep=False), "graph_id"].head(10).tolist()
        raise ValueError(f"[{dataset_name}] duplicate graph_id values: {duplicates}")

    if (df["speedup"] <= 0.0).any():
        bad_ids = df.loc[df["speedup"] <= 0.0, "graph_id"].head(10).tolist()
        raise ValueError(f"[{dataset_name}] non-positive speedup values at: {bad_ids}")

    if (df[["exact_time_ms", "sampled_time_ms"]] <= 0.0).any().any():
        bad_ids = df.loc[
            (df["exact_time_ms"] <= 0.0) | (df["sampled_time_ms"] <= 0.0),
            "graph_id",
        ].head(10).tolist()
        raise ValueError(f"[{dataset_name}] non-positive extraction times at: {bad_ids}")

    if graph_mode == "swap":
        df["base_graph"] = df["graph_id"].apply(extract_base_graph_id_from_swap)
    elif graph_mode == "sparse":
        df["base_graph"] = df["graph_id"].astype(str)
    elif graph_mode == "rmat":
        # RMAT IDs such as 1..75 are not the same entities as SuiteSparse IDs 1..75.
        df["base_graph"] = "rmat_" + df["graph_id"].astype(str)
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")

    df["kernel"] = kernel
    df["dataset"] = dataset_name

    for variant in FEATURE_VARIANTS:
        cols = source_r_cols(variant)
        values = df[cols].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"[{dataset_name}/{variant}] non-finite RD features found")
        if (values < -EPS).any():
            raise ValueError(f"[{dataset_name}/{variant}] negative RD probabilities found")

        row_sums = values.sum(axis=1)
        max_deviation = float(np.max(np.abs(row_sums - 1.0)))
        if max_deviation > ROW_SUM_WARN_TOL:
            print(
                f"WARNING [{dataset_name}/{variant}] max |sum(r)-1| = "
                f"{max_deviation:.6g} > {ROW_SUM_WARN_TOL:.6g}",
                file=sys.stderr,
            )

    print(f"\n[{kernel}/{dataset_name}]")
    print(f"  source: {path}")
    print(f"  rows: {len(df)}")
    print(f"  unique graph_id: {df['graph_id'].nunique()}")
    print(f"  unique base_graph: {df['base_graph'].nunique()}")
    print(f"  speedup range: [{df['speedup'].min():.6g}, {df['speedup'].max():.6g}]")
    for variant in FEATURE_VARIANTS:
        row_sums = df[source_r_cols(variant)].sum(axis=1)
        print(
            f"  {variant}: max |sum(r)-1|={np.max(np.abs(row_sums - 1.0)):.6g}, "
            f"median time={df[f'{variant}_time_ms'].median():.6g} ms"
        )

    return df


def make_feature_view(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Create a canonical r1...r10 view from exact_* or sampled_* columns."""
    src_cols = source_r_cols(variant)
    rename_map = {src: dst for src, dst in zip(src_cols, canonical_r_cols())}

    out = df[
        [
            "graph_id",
            "base_graph",
            "kernel",
            "dataset",
            "speedup",
            "exact_time_ms",
            "sampled_time_ms",
            *src_cols,
        ]
    ].copy()
    out = out.rename(columns=rename_map)
    out["feature_variant"] = variant
    return out


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    y = df["speedup"].to_numpy(dtype=float)
    rows: List[Dict[str, object]] = []

    for variant in FEATURE_VARIANTS:
        feature_values = df[source_r_cols(variant)].to_numpy(dtype=float)
        row_sum_deviation = np.abs(feature_values.sum(axis=1) - 1.0)
        time_values = df[f"{variant}_time_ms"].to_numpy(dtype=float)
        rows.append(
            {
                "kernel": str(df["kernel"].iloc[0]),
                "dataset": str(df["dataset"].iloc[0]),
                "feature_variant": variant,
                "rows": int(len(df)),
                "graph_ids": int(df["graph_id"].nunique()),
                "base_graphs": int(df["base_graph"].nunique()),
                "mean_speedup": float(np.mean(y)),
                "median_speedup": float(np.median(y)),
                "min_speedup": float(np.min(y)),
                "max_speedup": float(np.max(y)),
                "num_speedup_gt1": int(np.sum(y > 1.0)),
                "max_abs_feature_sum_deviation": float(np.max(row_sum_deviation)),
                "mean_extraction_time_ms": float(np.mean(time_values)),
                "median_extraction_time_ms": float(np.median(time_values)),
                "p90_extraction_time_ms": float(np.quantile(time_values, 0.90)),
            }
        )

    return pd.DataFrame(rows)


def validate_group_structure(
    train_df: pd.DataFrame,
    sparse_df: pd.DataFrame,
    expected_augmented_per_graph: Optional[int],
    out_dir: Path,
) -> None:
    train_counts = train_df.groupby("base_graph", sort=True).size()
    sparse_counts = sparse_df.groupby("base_graph", sort=True).size()

    pd.DataFrame(
        {"base_graph": train_counts.index, "num_augmented_samples": train_counts.values}
    ).to_csv(out_dir / "train_group_counts.csv", index=False)
    pd.DataFrame(
        {"base_graph": sparse_counts.index, "num_original_samples": sparse_counts.values}
    ).to_csv(out_dir / "sparse_test_group_counts.csv", index=False)

    if expected_augmented_per_graph is not None:
        bad = train_counts[train_counts != expected_augmented_per_graph]
        if len(bad) > 0:
            raise ValueError(
                f"Expected {expected_augmented_per_graph} augmented samples per base graph, "
                f"but {len(bad)} groups differ.\n{bad.to_string()}"
            )

    train_graphs = set(train_df["base_graph"].astype(str))
    sparse_graphs = set(sparse_df["base_graph"].astype(str))
    common = sorted(train_graphs & sparse_graphs, key=natural_graph_key)
    train_only = sorted(train_graphs - sparse_graphs, key=natural_graph_key)
    sparse_only = sorted(sparse_graphs - train_graphs, key=natural_graph_key)

    overlap_rows = [
        *({"base_graph": graph, "status": "common"} for graph in common),
        *({"base_graph": graph, "status": "train_only"} for graph in train_only),
        *({"base_graph": graph, "status": "sparse_only"} for graph in sparse_only),
    ]
    pd.DataFrame(overlap_rows).to_csv(out_dir / "train_sparse_graph_overlap.csv", index=False)

    if not common:
        raise ValueError("No common base graphs between augmented train and sparse original test")

    print("\n[graph structure]")
    print(f"  common SuiteSparse graphs: {len(common)}")
    print(f"  train-only graphs: {len(train_only)}")
    print(f"  sparse-only graphs: {len(sparse_only)}")


# =========================================================
# Exact-vs-sampled feature analysis
# =========================================================


def analyze_feature_approximation(df: pd.DataFrame, out_dir: Path) -> Dict[str, pd.DataFrame]:
    kernel = str(df["kernel"].iloc[0])
    dataset_name = str(df["dataset"].iloc[0])
    exact = df[source_r_cols("exact")].to_numpy(dtype=float)
    sampled = df[source_r_cols("sampled")].to_numpy(dtype=float)
    difference = sampled - exact
    absolute_difference = np.abs(difference)

    l1 = np.sum(absolute_difference, axis=1)
    tv = 0.5 * l1
    l2 = np.sqrt(np.sum(difference * difference, axis=1))
    cosine_denominator = np.linalg.norm(exact, axis=1) * np.linalg.norm(sampled, axis=1)
    cosine = np.divide(
        np.sum(exact * sampled, axis=1),
        cosine_denominator,
        out=np.full(len(df), np.nan),
        where=cosine_denominator > EPS,
    )

    exact_time = df["exact_time_ms"].to_numpy(dtype=float)
    sampled_time = df["sampled_time_ms"].to_numpy(dtype=float)
    time_speedup = exact_time / sampled_time

    per_row = pd.DataFrame(
        {
            "kernel": kernel,
            "dataset": dataset_name,
            "graph_id": df["graph_id"].astype(str).to_numpy(),
            "base_graph": df["base_graph"].astype(str).to_numpy(),
            "feature_l1": l1,
            "total_variation": tv,
            "feature_l2": l2,
            "cosine_similarity": cosine,
            "max_abs_bin_error": np.max(absolute_difference, axis=1),
            "exact_time_ms": exact_time,
            "sampled_time_ms": sampled_time,
            "extraction_speedup_exact_over_sampled": time_speedup,
        }
    )
    for index in range(FEATURE_COUNT):
        per_row[f"exact_r{index + 1}"] = exact[:, index]
        per_row[f"sampled_r{index + 1}"] = sampled[:, index]
        per_row[f"sample_minus_exact_r{index + 1}"] = difference[:, index]

    per_row.to_csv(out_dir / f"{dataset_name}_feature_approximation_per_row.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "kernel": kernel,
                "dataset": dataset_name,
                "rows": int(len(df)),
                "mean_l1": float(np.mean(l1)),
                "median_l1": float(np.median(l1)),
                "p90_l1": float(np.quantile(l1, 0.90)),
                "p95_l1": float(np.quantile(l1, 0.95)),
                "max_l1": float(np.max(l1)),
                "mean_total_variation": float(np.mean(tv)),
                "mean_l2": float(np.mean(l2)),
                "mean_cosine_similarity": float(np.nanmean(cosine)),
                "mean_max_abs_bin_error": float(np.mean(np.max(absolute_difference, axis=1))),
                "mean_exact_time_ms": float(np.mean(exact_time)),
                "median_exact_time_ms": float(np.median(exact_time)),
                "mean_sampled_time_ms": float(np.mean(sampled_time)),
                "median_sampled_time_ms": float(np.median(sampled_time)),
                "mean_extraction_speedup": float(np.mean(time_speedup)),
                "geomean_extraction_speedup": float(np.exp(np.mean(np.log(time_speedup)))),
                "median_extraction_speedup": float(np.median(time_speedup)),
                "p10_extraction_speedup": float(np.quantile(time_speedup, 0.10)),
                "p90_extraction_speedup": float(np.quantile(time_speedup, 0.90)),
            }
        ]
    )
    summary.to_csv(out_dir / f"{dataset_name}_feature_approximation_summary.csv", index=False)

    bin_rows: List[Dict[str, object]] = []
    for index in range(FEATURE_COUNT):
        e = exact[:, index]
        s = sampled[:, index]
        d = s - e
        bin_rows.append(
            {
                "kernel": kernel,
                "dataset": dataset_name,
                "feature": f"r{index + 1}",
                "exact_mean": float(np.mean(e)),
                "sampled_mean": float(np.mean(s)),
                "bias_sample_minus_exact": float(np.mean(d)),
                "mae": float(np.mean(np.abs(d))),
                "rmse": float(np.sqrt(np.mean(d * d))),
                "max_abs_error": float(np.max(np.abs(d))),
                "pearson": pearson_corr(e, s),
                "spearman": spearman_corr(e, s),
            }
        )
    per_bin = pd.DataFrame(bin_rows)
    per_bin.to_csv(out_dir / f"{dataset_name}_feature_approximation_per_bin.csv", index=False)

    return {"summary": summary, "per_row": per_row, "per_bin": per_bin}


# =========================================================
# Model and deployment export
# =========================================================


def build_preprocess(r_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                r_cols,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(r_cols: List[str], ridge_alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocess(r_cols)),
            ("model", Ridge(alpha=ridge_alpha)),
        ]
    )


def extract_model_parameters(pipe: Pipeline, r_cols: List[str]) -> Dict[str, object]:
    preprocess = pipe.named_steps["preprocess"]
    ridge = pipe.named_steps["model"]
    numeric_pipeline = preprocess.named_transformers_["num"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]

    imputer_median = np.asarray(imputer.statistics_, dtype=np.float64).reshape(-1)
    scaler_mean = np.asarray(scaler.mean_, dtype=np.float64).reshape(-1)
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)
    coef_scaled = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
    intercept_scaled = float(np.asarray(ridge.intercept_, dtype=np.float64).reshape(()))

    expected = len(r_cols)
    for name, values in {
        "imputer_median": imputer_median,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "coef_scaled": coef_scaled,
    }.items():
        if len(values) != expected:
            raise ValueError(f"{name} length {len(values)} != feature count {expected}")

    if np.any(~np.isfinite(scaler_scale)) or np.any(scaler_scale <= 0.0):
        raise ValueError(f"Invalid StandardScaler scales: {scaler_scale}")

    coef_raw = coef_scaled / scaler_scale
    intercept_raw = intercept_scaled - float(
        np.dot(coef_scaled, scaler_mean / scaler_scale)
    )

    return {
        "feature_order": list(r_cols),
        "imputer_median": imputer_median,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "coef_scaled": coef_scaled,
        "intercept_scaled": intercept_scaled,
        "coef_raw": coef_raw,
        "intercept_raw": intercept_raw,
        "ridge_alpha": float(ridge.alpha),
    }


def model_parameter_dataframe(
    pipe: Pipeline,
    r_cols: List[str],
    variant: str,
    kernel: str,
) -> pd.DataFrame:
    params = extract_model_parameters(pipe, r_cols)
    out = pd.DataFrame(
        {
            "kernel": kernel,
            "feature_variant": variant,
            "feature": params["feature_order"],
            "imputer_median": params["imputer_median"],
            "scaler_mean": params["scaler_mean"],
            "scaler_scale": params["scaler_scale"],
            "ridge_coef_scaled": params["coef_scaled"],
            "abs_ridge_coef_scaled": np.abs(params["coef_scaled"]),
            "ridge_coef_raw": params["coef_raw"],
            "abs_ridge_coef_raw": np.abs(params["coef_raw"]),
        }
    )
    out["ridge_intercept_scaled"] = params["intercept_scaled"]
    out["ridge_intercept_raw"] = params["intercept_raw"]
    out["ridge_alpha"] = params["ridge_alpha"]
    out["target"] = "log(speedup)"
    return out


def _cpp_array(values: Sequence[float], indent: str = "        ") -> str:
    return (",\n" + indent).join(f"{float(value):.17e}" for value in values)


def export_deployment_model(
    pipe: Pipeline,
    r_cols: List[str],
    variant: str,
    kernel: str,
    out_dir: Path,
    verification_frames: Mapping[str, pd.DataFrame],
) -> Dict[str, Path]:
    params = extract_model_parameters(pipe, r_cols)
    prefix = f"{kernel}_{variant}_deploy_train_all_aug"

    csv_path = out_dir / f"{prefix}_parameters.csv"
    json_path = out_dir / f"{prefix}_model.json"
    hpp_path = out_dir / f"rd_{variant}_{kernel}_model.hpp"
    verification_path = out_dir / f"{prefix}_verification.csv"

    model_parameter_dataframe(pipe, r_cols, variant, kernel).to_csv(csv_path, index=False)

    payload = {
        "kernel": kernel,
        "feature_variant": variant,
        "source_columns": source_r_cols(variant),
        "canonical_feature_order": params["feature_order"],
        "model_type": "Ridge",
        "ridge_alpha": params["ridge_alpha"],
        "training_target": "log(speedup)",
        "inverse_transform": "exp(predicted_log_speedup)",
        "preprocessing": {
            "imputer_strategy": "median",
            "imputer_statistics": [float(x) for x in params["imputer_median"]],
            "standard_scaler_mean": [float(x) for x in params["scaler_mean"]],
            "standard_scaler_scale": [float(x) for x in params["scaler_scale"]],
        },
        "ridge_standardized": {
            "intercept": params["intercept_scaled"],
            "coefficients": [float(x) for x in params["coef_scaled"]],
        },
        "ridge_raw_finite_features": {
            "intercept": params["intercept_raw"],
            "coefficients": [float(x) for x in params["coef_raw"]],
            "formula": "speedup = exp(intercept + sum(coef[i] * r[i]))",
        },
    }
    write_json(json_path, payload)

    namespace = f"rd_{variant}_{kernel}_model"
    feature_names = ",\n        ".join(f'"{name}"' for name in params["feature_order"])
    header = f"""#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace {namespace}
{{
    inline constexpr std::size_t kFeatureCount = {len(r_cols)};

    // Input order must be r1, r2, ..., r10 from the {variant} RD extractor.
    inline constexpr std::array<const char*, kFeatureCount> kFeatureNames = {{
        {feature_names}
    }};

    inline constexpr std::array<double, kFeatureCount> kImputerMedian = {{
        {_cpp_array(params["imputer_median"])}
    }};

    inline constexpr std::array<double, kFeatureCount> kScalerMean = {{
        {_cpp_array(params["scaler_mean"])}
    }};

    inline constexpr std::array<double, kFeatureCount> kScalerScale = {{
        {_cpp_array(params["scaler_scale"])}
    }};

    inline constexpr std::array<double, kFeatureCount> kScaledCoef = {{
        {_cpp_array(params["coef_scaled"])}
    }};

    inline constexpr double kScaledIntercept =
        {params["intercept_scaled"]:.17e};

    // StandardScaler has been folded into these parameters.
    inline constexpr std::array<double, kFeatureCount> kRawCoef = {{
        {_cpp_array(params["coef_raw"])}
    }};

    inline constexpr double kRawIntercept =
        {params["intercept_raw"]:.17e};

    inline double PredictLogSpeedup(
        const std::array<double, kFeatureCount>& rd_feature)
    {{
        double value = kRawIntercept;
        for(std::size_t i = 0; i < kFeatureCount; ++i)
        {{
            value += kRawCoef[i] * rd_feature[i];
        }}
        return value;
    }}

    inline double PredictSpeedup(
        const std::array<double, kFeatureCount>& rd_feature)
    {{
        return std::exp(PredictLogSpeedup(rd_feature));
    }}

    inline double PredictSpeedupWithPreprocess(
        std::array<double, kFeatureCount> rd_feature,
        const std::array<bool, kFeatureCount>& missing)
    {{
        double value = kScaledIntercept;
        for(std::size_t i = 0; i < kFeatureCount; ++i)
        {{
            const double x = missing[i] ? kImputerMedian[i] : rd_feature[i];
            const double standardized = (x - kScalerMean[i]) / kScalerScale[i];
            value += kScaledCoef[i] * standardized;
        }}
        return std::exp(value);
    }}
}}
"""
    hpp_path.write_text(header, encoding="utf-8")

    verification_parts: List[pd.DataFrame] = []
    for dataset_name, frame in verification_frames.items():
        if len(frame) == 0:
            continue
        x = frame[r_cols].copy()
        sklearn_log = np.asarray(pipe.predict(x), dtype=np.float64)
        x_raw = x.to_numpy(dtype=np.float64)
        raw_log = params["intercept_raw"] + x_raw @ params["coef_raw"]
        part = pd.DataFrame(
            {
                "dataset": dataset_name,
                "graph_id": frame["graph_id"].astype(str).to_numpy(),
                "sklearn_pred_log_speedup": sklearn_log,
                "raw_formula_pred_log_speedup": raw_log,
                "abs_log_difference": np.abs(sklearn_log - raw_log),
                "sklearn_pred_speedup": np.exp(sklearn_log),
                "raw_formula_pred_speedup": np.exp(raw_log),
                "abs_speedup_difference": np.abs(np.exp(sklearn_log) - np.exp(raw_log)),
            }
        )
        verification_parts.append(part)

    verification_df = pd.concat(verification_parts, ignore_index=True)
    verification_df.to_csv(verification_path, index=False)

    max_log_diff = float(verification_df["abs_log_difference"].max())
    max_speedup_diff = float(verification_df["abs_speedup_difference"].max())
    if max_log_diff > 1e-10 or max_speedup_diff > 1e-10:
        raise RuntimeError(
            f"[{variant}] exported raw model does not reproduce sklearn: "
            f"max log diff={max_log_diff}, max speedup diff={max_speedup_diff}"
        )

    return {
        "csv": csv_path,
        "json": json_path,
        "hpp": hpp_path,
        "verification": verification_path,
    }


# =========================================================
# Splits and evaluation
# =========================================================


def build_row_group_splits(df: pd.DataFrame, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = df["base_graph"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"n_splits={n_splits} > unique groups={len(unique_groups)}")

    splitter = GroupKFold(n_splits=n_splits)
    x_dummy = np.zeros((len(df), 1), dtype=float)
    y_dummy = np.zeros(len(df), dtype=float)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(x_dummy, y_dummy, groups)]


def build_graph_folds(common_graphs: Sequence[str], n_splits: int) -> List[GraphFold]:
    graphs = np.asarray(sorted(set(map(str, common_graphs)), key=natural_graph_key), dtype=object)
    if len(graphs) < n_splits:
        raise ValueError(f"n_splits={n_splits} > common graphs={len(graphs)}")

    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(graphs), 1), dtype=float)
    dummy_y = np.zeros(len(graphs), dtype=float)

    folds: List[GraphFold] = []
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(dummy_x, dummy_y, groups=graphs), start=1
    ):
        train_graphs = tuple(str(x) for x in graphs[train_idx])
        test_graphs = tuple(str(x) for x in graphs[test_idx])
        if set(train_graphs) & set(test_graphs):
            raise RuntimeError(f"Graph leakage in fold {fold_id}")
        folds.append(GraphFold(fold_id, train_graphs, test_graphs))
    return folds


def save_graph_folds(folds: Sequence[GraphFold], path: Path) -> None:
    rows: List[Dict[str, object]] = []
    for fold in folds:
        rows.extend(
            {"fold": fold.fold, "base_graph": graph, "role": "train"}
            for graph in fold.train_graphs
        )
        rows.extend(
            {"fold": fold.fold, "base_graph": graph, "role": "test"}
            for graph in fold.test_graphs
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def evaluate_regression(y_true_log: Sequence[float], y_pred_log: Sequence[float]) -> Dict[str, float]:
    true_log = np.asarray(y_true_log, dtype=float)
    pred_log = np.asarray(y_pred_log, dtype=float)
    true_speedup = np.exp(true_log)
    pred_speedup = np.exp(pred_log)

    return {
        "mae_log": float(mean_absolute_error(true_log, pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(true_log, pred_log))),
        "r2_log": float(r2_score(true_log, pred_log)) if len(true_log) >= 2 else np.nan,
        "spearman": spearman_corr(true_speedup, pred_speedup),
        "pearson_log": pearson_corr(true_log, pred_log),
        "mae_speedup": float(mean_absolute_error(true_speedup, pred_speedup)),
        "rmse_speedup": float(np.sqrt(mean_squared_error(true_speedup, pred_speedup))),
        "mape": mape(true_speedup, pred_speedup),
        "smape": symmetric_mape(true_speedup, pred_speedup),
    }


def prediction_dataframe(
    source_df: pd.DataFrame,
    pred_log: Sequence[float],
    variant: str,
    experiment: str,
    fold: Optional[int] = None,
) -> pd.DataFrame:
    true_speedup = source_df["speedup"].to_numpy(dtype=float)
    true_log = np.log(true_speedup)
    pred_log_arr = np.asarray(pred_log, dtype=float)
    pred_speedup = np.exp(pred_log_arr)

    out = source_df[
        ["graph_id", "base_graph", "kernel", "dataset", "speedup", *canonical_r_cols()]
    ].copy()
    out["feature_variant"] = variant
    out["experiment"] = experiment
    if fold is not None:
        out["fold"] = fold
    out["true_log_speedup"] = true_log
    out["pred_log_speedup"] = pred_log_arr
    out["pred_speedup"] = pred_speedup
    out["signed_error_speedup"] = pred_speedup - true_speedup
    out["abs_error_speedup"] = np.abs(pred_speedup - true_speedup)
    out["ape"] = out["abs_error_speedup"] / np.maximum(np.abs(true_speedup), EPS)
    out["true_beneficial"] = true_speedup > 1.0
    return out


def fit_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str,
    ridge_alpha: float,
    experiment: str,
    fold: Optional[int] = None,
) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame]:
    r_cols = canonical_r_cols()
    pipe = make_pipeline(r_cols, ridge_alpha)
    y_train_log = np.log(train_df["speedup"].to_numpy(dtype=float))
    pipe.fit(train_df[r_cols], y_train_log)

    pred_log = pipe.predict(test_df[r_cols])
    metrics = evaluate_regression(np.log(test_df["speedup"].to_numpy(dtype=float)), pred_log)
    pred_df = prediction_dataframe(test_df, pred_log, variant, experiment, fold)
    return pipe, metrics, pred_df


def run_band_gate_scan(
    pred_df: pd.DataFrame,
    tau_pairs: Iterable[Tuple[float, float]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    true_speedup = pred_df["speedup"].to_numpy(dtype=float)
    pred_speedup = pred_df["pred_speedup"].to_numpy(dtype=float)
    true_positive = true_speedup > 1.0

    for tau_l, tau_h in tau_pairs:
        decision = np.full(len(pred_df), "uncertain", dtype=object)
        decision[pred_speedup >= tau_h] = "relabel"
        decision[pred_speedup <= tau_l] = "skip"
        decided = decision != "uncertain"

        predicted_positive = decision[decided] == "relabel"
        actual_positive = true_positive[decided]

        tp = int(np.sum(predicted_positive & actual_positive))
        tn = int(np.sum((~predicted_positive) & (~actual_positive)))
        fp = int(np.sum(predicted_positive & (~actual_positive)))
        fn = int(np.sum((~predicted_positive) & actual_positive))
        count = tp + tn + fp + fn

        rows.append(
            {
                "tau_l": tau_l,
                "tau_h": tau_h,
                "coverage": float(np.mean(decided)) if len(decided) else np.nan,
                "accuracy": (tp + tn) / count if count else np.nan,
                "precision": tp / (tp + fp) if (tp + fp) else np.nan,
                "recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "num_decided": int(np.sum(decided)),
                "num_total": int(len(pred_df)),
            }
        )

    return pd.DataFrame(rows)


def run_conservative_gate(
    pred_df: pd.DataFrame,
    tau_h_values: Iterable[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    true_speedup = pred_df["speedup"].to_numpy(dtype=float)
    pred_speedup = pred_df["pred_speedup"].to_numpy(dtype=float)
    true_positive = true_speedup > 1.0

    for tau_h in tau_h_values:
        relabel = pred_speedup >= tau_h
        effective_speedup = np.where(relabel, true_speedup, 1.0)
        selected = int(np.sum(relabel))
        good_selected = int(np.sum(relabel & true_positive))
        bad_selected = int(np.sum(relabel & (~true_positive)))
        missed_good = int(np.sum((~relabel) & true_positive))

        rows.append(
            {
                "tau_h": tau_h,
                "selected_relabel": selected,
                "selection_rate": selected / len(pred_df) if len(pred_df) else np.nan,
                "good_selected": good_selected,
                "bad_selected_speedup_le1": bad_selected,
                "selection_precision": good_selected / selected if selected else np.nan,
                "missed_good_speedup_gt1": missed_good,
                "mean_effective_speedup": float(np.mean(effective_speedup)),
                "median_effective_speedup": float(np.median(effective_speedup)),
                "geomean_effective_speedup": float(np.exp(np.mean(np.log(effective_speedup)))),
                "always_relabel_mean_speedup": float(np.mean(true_speedup)),
                "oracle_mean_effective_speedup": float(
                    np.mean(np.where(true_positive, true_speedup, 1.0))
                ),
            }
        )

    return pd.DataFrame(rows)


def save_gate_tables(pred_df: pd.DataFrame, prefix: Path) -> Dict[str, pd.DataFrame]:
    band = run_band_gate_scan(pred_df, BAND_GATE_PAIRS)
    conservative = run_conservative_gate(pred_df, [pair[1] for pair in BAND_GATE_PAIRS])
    band.to_csv(f"{prefix}_band_gate.csv", index=False)
    conservative.to_csv(f"{prefix}_conservative_gate.csv", index=False)
    return {"band": band, "conservative": conservative}


def run_augmented_group_cv(
    train_df: pd.DataFrame,
    variant: str,
    graph_folds: Sequence[GraphFold],
    ridge_alpha: float,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Grouped CV on augmented rows using the shared graph folds."""
    experiment = "A_group_cv_augmented"
    pred_parts: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, object]] = []

    for fold in graph_folds:
        train_part = train_df[train_df["base_graph"].isin(fold.train_graphs)]
        valid_part = train_df[train_df["base_graph"].isin(fold.test_graphs)]

        if len(train_part) == 0 or len(valid_part) == 0:
            raise RuntimeError(
                f"[{train_df['kernel'].iloc[0]}/{variant}] empty augmented fold {fold.fold}"
            )

        train_groups = set(train_part["base_graph"].astype(str))
        valid_groups = set(valid_part["base_graph"].astype(str))
        if train_groups & valid_groups:
            raise RuntimeError(
                f"[{train_df['kernel'].iloc[0]}/{variant}] graph leakage in fold {fold.fold}"
            )

        _, metrics, pred_df = fit_and_predict(
            train_part,
            valid_part,
            variant,
            ridge_alpha,
            experiment,
            fold.fold,
        )
        pred_parts.append(pred_df)
        fold_rows.append(
            {
                "kernel": str(train_df["kernel"].iloc[0]),
                "feature_variant": variant,
                "experiment": experiment,
                "fold": fold.fold,
                "train_rows": int(len(train_part)),
                "valid_rows": int(len(valid_part)),
                "train_base_graphs": int(len(train_groups)),
                "valid_base_graphs": int(len(valid_groups)),
                **metrics,
            }
        )

    oof_df = pd.concat(pred_parts, ignore_index=True)
    if len(oof_df) != len(train_df):
        raise RuntimeError(
            f"[{train_df['kernel'].iloc[0]}/{variant}] augmented OOF coverage "
            f"{len(oof_df)} != {len(train_df)}"
        )
    if oof_df["graph_id"].duplicated().any():
        raise RuntimeError(
            f"[{train_df['kernel'].iloc[0]}/{variant}] duplicate augmented OOF predictions"
        )

    oof_metrics = evaluate_regression(
        oof_df["true_log_speedup"], oof_df["pred_log_speedup"]
    )
    folds_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(
        [
            {
                "kernel": str(train_df["kernel"].iloc[0]),
                "feature_variant": variant,
                "experiment": experiment,
                "split_type": "Shared GroupKFold by base graph on augmented samples",
                "cv_folds": int(len(graph_folds)),
                "oof_rows": int(len(oof_df)),
                **{f"oof_{key}": value for key, value in oof_metrics.items()},
                "fold_mae_log_mean": float(folds_df["mae_log"].mean()),
                "fold_mae_log_std": float(folds_df["mae_log"].std(ddof=1)),
                "fold_spearman_mean": float(folds_df["spearman"].mean()),
                "fold_spearman_std": float(folds_df["spearman"].std(ddof=1)),
                "fold_mape_mean": float(folds_df["mape"].mean()),
                "fold_mape_std": float(folds_df["mape"].std(ddof=1)),
            }
        ]
    )

    folds_df.to_csv(out_dir / f"{experiment}_folds.csv", index=False)
    summary_df.to_csv(out_dir / f"{experiment}_summary.csv", index=False)
    oof_df.to_csv(out_dir / f"{experiment}_oof_predictions.csv", index=False)
    save_gate_tables(oof_df, out_dir / f"{experiment}_oof")
    return summary_df, folds_df, oof_df

def run_heldout_sparse_graphs(
    train_df: pd.DataFrame,
    sparse_df: pd.DataFrame,
    variant: str,
    graph_folds: Sequence[GraphFold],
    ridge_alpha: float,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiment = "C_heldout_sparse_graphs"
    pred_parts: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, object]] = []

    for fold in graph_folds:
        train_part = train_df[train_df["base_graph"].isin(fold.train_graphs)]
        test_part = sparse_df[sparse_df["base_graph"].isin(fold.test_graphs)]

        if len(train_part) == 0 or len(test_part) == 0:
            raise RuntimeError(
                f"[{train_df['kernel'].iloc[0]}/{variant}] empty held-out fold {fold.fold}"
            )

        _, metrics, pred_df = fit_and_predict(
            train_part,
            test_part,
            variant,
            ridge_alpha,
            experiment,
            fold.fold,
        )
        pred_parts.append(pred_df)
        fold_rows.append(
            {
                "kernel": str(train_df["kernel"].iloc[0]),
                "feature_variant": variant,
                "experiment": experiment,
                "fold": fold.fold,
                "train_augmented_rows": int(len(train_part)),
                "test_original_rows": int(len(test_part)),
                "train_base_graphs": int(len(fold.train_graphs)),
                "test_base_graphs": int(len(fold.test_graphs)),
                **metrics,
            }
        )

    oof_df = pd.concat(pred_parts, ignore_index=True)
    if oof_df["graph_id"].duplicated().any():
        raise RuntimeError(f"[{train_df['kernel'].iloc[0]}/{variant}] duplicate held-out predictions")
    if len(oof_df) != len(sparse_df):
        raise RuntimeError(
            f"[{train_df['kernel'].iloc[0]}/{variant}] held-out predictions cover "
            f"{len(oof_df)} rows, expected {len(sparse_df)}"
        )

    overall = evaluate_regression(
        oof_df["true_log_speedup"], oof_df["pred_log_speedup"]
    )
    folds_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(
        [
            {
                "kernel": str(train_df["kernel"].iloc[0]),
                "feature_variant": variant,
                "experiment": experiment,
                "split_type": (
                    "Train augmented samples from training graphs; "
                    "test original held-out graphs"
                ),
                "cv_folds": int(len(graph_folds)),
                "oof_test_rows": int(len(oof_df)),
                **{f"oof_{key}": value for key, value in overall.items()},
                "fold_mae_log_mean": float(folds_df["mae_log"].mean()),
                "fold_mae_log_std": float(folds_df["mae_log"].std(ddof=1)),
                "fold_spearman_mean": float(folds_df["spearman"].mean()),
                "fold_spearman_std": float(folds_df["spearman"].std(ddof=1)),
                "fold_mape_mean": float(folds_df["mape"].mean()),
                "fold_mape_std": float(folds_df["mape"].std(ddof=1)),
            }
        ]
    )

    folds_df.to_csv(out_dir / f"{experiment}_folds.csv", index=False)
    summary_df.to_csv(out_dir / f"{experiment}_summary.csv", index=False)
    oof_df.to_csv(out_dir / f"{experiment}_oof_predictions.csv", index=False)
    save_gate_tables(oof_df, out_dir / f"{experiment}_oof")
    return summary_df, folds_df, oof_df

def run_single_fit_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str,
    ridge_alpha: float,
    experiment: str,
    description: str,
    out_dir: Path,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    pipe, metrics, pred_df = fit_and_predict(
        train_df,
        test_df,
        variant,
        ridge_alpha,
        experiment,
    )
    summary = pd.DataFrame(
        [
            {
                "kernel": str(train_df["kernel"].iloc[0]),
                "feature_variant": variant,
                "experiment": experiment,
                "description": description,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_base_graphs": int(train_df["base_graph"].nunique()),
                "test_base_graphs": int(test_df["base_graph"].nunique()),
                **metrics,
            }
        ]
    )
    pred_df.to_csv(out_dir / f"{experiment}_predictions.csv", index=False)
    summary.to_csv(out_dir / f"{experiment}_summary.csv", index=False)
    gate_tables = save_gate_tables(pred_df, out_dir / experiment)
    return pipe, summary, pred_df, gate_tables

def run_variant(
    kernel: str,
    variant: str,
    train_df_dual: pd.DataFrame,
    sparse_df_dual: pd.DataFrame,
    rmat_df_dual: pd.DataFrame,
    graph_folds: Sequence[GraphFold],
    ridge_alpha: float,
    root_out_dir: Path,
) -> VariantResult:
    variant_dir = ensure_directory(root_out_dir / variant)
    train_df = make_feature_view(train_df_dual, variant)
    sparse_df = make_feature_view(sparse_df_dual, variant)
    rmat_df = make_feature_view(rmat_df_dual, variant)

    exp_a_summary, _, pred_a = run_augmented_group_cv(
        train_df,
        variant,
        graph_folds,
        ridge_alpha,
        variant_dir,
    )

    model_all_aug, exp_b_summary, pred_b, gate_b = run_single_fit_experiment(
        train_df,
        sparse_df,
        variant,
        ridge_alpha,
        "B_same_graph_sparse",
        (
            "Train on all augmented SuiteSparse samples and test original "
            "samples from the same graph set."
        ),
        variant_dir,
    )

    exp_c_summary, _, pred_c = run_heldout_sparse_graphs(
        train_df,
        sparse_df,
        variant,
        graph_folds,
        ridge_alpha,
        variant_dir,
    )

    _, exp_d_summary, pred_d, gate_d = run_single_fit_experiment(
        train_df,
        rmat_df,
        variant,
        ridge_alpha,
        "D_external_rmat",
        "Train on all augmented SuiteSparse samples and test external RMAT graphs.",
        variant_dir,
    )

    coefficient_table = model_parameter_dataframe(
        model_all_aug, canonical_r_cols(), variant, kernel
    )
    coefficient_table.to_csv(variant_dir / "deploy_model_parameters.csv", index=False)
    coefficient_table.sort_values("abs_ridge_coef_scaled", ascending=False).to_csv(
        variant_dir / "deploy_model_coefficients_by_importance.csv", index=False
    )

    export_deployment_model(
        model_all_aug,
        canonical_r_cols(),
        variant,
        kernel,
        variant_dir,
        verification_frames={"sparse": sparse_df, "rmat": rmat_df},
    )

    summary_columns = [
        "kernel",
        "feature_variant",
        "experiment",
        "mae_log",
        "rmse_log",
        "r2_log",
        "spearman",
        "pearson_log",
        "mae_speedup",
        "rmse_speedup",
        "mape",
        "smape",
    ]

    exp_a_main = exp_a_summary.rename(
        columns={
            "oof_mae_log": "mae_log",
            "oof_rmse_log": "rmse_log",
            "oof_r2_log": "r2_log",
            "oof_spearman": "spearman",
            "oof_pearson_log": "pearson_log",
            "oof_mae_speedup": "mae_speedup",
            "oof_rmse_speedup": "rmse_speedup",
            "oof_mape": "mape",
            "oof_smape": "smape",
        }
    )[summary_columns]
    exp_c_main = exp_c_summary.rename(
        columns={
            "oof_mae_log": "mae_log",
            "oof_rmse_log": "rmse_log",
            "oof_r2_log": "r2_log",
            "oof_spearman": "spearman",
            "oof_pearson_log": "pearson_log",
            "oof_mae_speedup": "mae_speedup",
            "oof_rmse_speedup": "rmse_speedup",
            "oof_mape": "mape",
            "oof_smape": "smape",
        }
    )[summary_columns]

    master = pd.concat(
        [
            exp_a_main,
            exp_b_summary[summary_columns],
            exp_c_main,
            exp_d_summary[summary_columns],
        ],
        ignore_index=True,
    )
    master["recommended_for_main_table"] = master["experiment"].isin(
        ["C_heldout_sparse_graphs", "D_external_rmat"]
    )
    master.to_csv(variant_dir / "master_summary.csv", index=False)

    return VariantResult(
        kernel=kernel,
        variant=variant,
        model_all_aug=model_all_aug,
        feature_cols=canonical_r_cols(),
        master_summary=master,
        predictions={
            "A_group_cv_augmented": pred_a,
            "B_same_graph_sparse": pred_b,
            "C_heldout_sparse_graphs": pred_c,
            "D_external_rmat": pred_d,
        },
        gate_tables={
            "B_same_graph_sparse_band": gate_b["band"],
            "B_same_graph_sparse_conservative": gate_b["conservative"],
            "D_external_rmat_band": gate_d["band"],
            "D_external_rmat_conservative": gate_d["conservative"],
        },
        coefficient_table=coefficient_table,
    )

def compare_prediction_frames(
    exact_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    experiment: str,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["kernel", "graph_id", "base_graph"]
    optional_keys = [
        column
        for column in ["fold"]
        if column in exact_df.columns and column in sampled_df.columns
    ]
    merge_keys = keys + optional_keys

    selected = [
        *merge_keys,
        "dataset",
        "speedup",
        "pred_speedup",
        "pred_log_speedup",
        "abs_error_speedup",
    ]
    e = exact_df[selected].rename(
        columns={
            "dataset": "dataset_exact",
            "speedup": "true_speedup_exact",
            "pred_speedup": "pred_speedup_exact",
            "pred_log_speedup": "pred_log_speedup_exact",
            "abs_error_speedup": "abs_error_speedup_exact",
        }
    )
    s = sampled_df[selected].rename(
        columns={
            "dataset": "dataset_sampled",
            "speedup": "true_speedup_sampled",
            "pred_speedup": "pred_speedup_sampled",
            "pred_log_speedup": "pred_log_speedup_sampled",
            "abs_error_speedup": "abs_error_speedup_sampled",
        }
    )

    paired = pd.merge(e, s, on=merge_keys, how="inner", validate="one_to_one")
    if len(paired) != len(exact_df) or len(paired) != len(sampled_df):
        raise RuntimeError(
            f"[{experiment}] exact/sample prediction alignment failed: "
            f"exact={len(exact_df)}, sampled={len(sampled_df)}, merged={len(paired)}"
        )

    if not np.allclose(
        paired["true_speedup_exact"],
        paired["true_speedup_sampled"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"[{experiment}] exact/sample true targets differ")

    paired["experiment"] = experiment
    paired["true_speedup"] = paired["true_speedup_exact"]
    paired["sampled_minus_exact_pred_speedup"] = (
        paired["pred_speedup_sampled"] - paired["pred_speedup_exact"]
    )
    paired["abs_prediction_difference"] = np.abs(
        paired["sampled_minus_exact_pred_speedup"]
    )
    paired["sampled_error_minus_exact_error"] = (
        paired["abs_error_speedup_sampled"] - paired["abs_error_speedup_exact"]
    )
    paired["sampled_prediction_closer"] = (
        paired["abs_error_speedup_sampled"] < paired["abs_error_speedup_exact"]
    )
    paired.to_csv(
        out_dir / f"{experiment}_exact_vs_sampled_predictions.csv", index=False
    )

    kernel = str(paired["kernel"].iloc[0])
    summary = pd.DataFrame(
        [
            {
                "kernel": kernel,
                "experiment": experiment,
                "rows": int(len(paired)),
                "mean_abs_exact_sampled_prediction_difference": float(
                    paired["abs_prediction_difference"].mean()
                ),
                "median_abs_exact_sampled_prediction_difference": float(
                    paired["abs_prediction_difference"].median()
                ),
                "max_abs_exact_sampled_prediction_difference": float(
                    paired["abs_prediction_difference"].max()
                ),
                "pearson_between_predictions": pearson_corr(
                    paired["pred_speedup_exact"], paired["pred_speedup_sampled"]
                ),
                "spearman_between_predictions": spearman_corr(
                    paired["pred_speedup_exact"], paired["pred_speedup_sampled"]
                ),
                "mean_abs_error_exact": float(
                    paired["abs_error_speedup_exact"].mean()
                ),
                "mean_abs_error_sampled": float(
                    paired["abs_error_speedup_sampled"].mean()
                ),
                "sampled_minus_exact_mean_abs_error": float(
                    paired["sampled_error_minus_exact_error"].mean()
                ),
                "num_sampled_closer": int(paired["sampled_prediction_closer"].sum()),
                "fraction_sampled_closer": float(
                    paired["sampled_prediction_closer"].mean()
                ),
            }
        ]
    )
    summary.to_csv(
        out_dir / f"{experiment}_exact_vs_sampled_summary.csv", index=False
    )

    disagreement_rows: List[Dict[str, object]] = []
    true_speedup = paired["true_speedup"].to_numpy(dtype=float)
    for tau_h in [pair[1] for pair in BAND_GATE_PAIRS]:
        exact_select = paired["pred_speedup_exact"].to_numpy(dtype=float) >= tau_h
        sampled_select = paired["pred_speedup_sampled"].to_numpy(dtype=float) >= tau_h
        disagreement = exact_select != sampled_select
        sampled_only = (~exact_select) & sampled_select
        exact_only = exact_select & (~sampled_select)

        disagreement_rows.append(
            {
                "kernel": kernel,
                "experiment": experiment,
                "tau_h": tau_h,
                "rows": int(len(paired)),
                "decision_disagreement_count": int(np.sum(disagreement)),
                "decision_disagreement_rate": float(np.mean(disagreement)),
                "sampled_only_select_count": int(np.sum(sampled_only)),
                "exact_only_select_count": int(np.sum(exact_only)),
                "sampled_only_bad_count": int(
                    np.sum(sampled_only & (true_speedup <= 1.0))
                ),
                "exact_only_bad_count": int(
                    np.sum(exact_only & (true_speedup <= 1.0))
                ),
            }
        )

    disagreement_df = pd.DataFrame(disagreement_rows)
    disagreement_df.to_csv(
        out_dir / f"{experiment}_gate_disagreement.csv", index=False
    )
    return paired, summary, disagreement_df

def compare_variants(
    exact_result: VariantResult,
    sampled_result: VariantResult,
    out_dir: Path,
) -> pd.DataFrame:
    if exact_result.kernel != sampled_result.kernel:
        raise ValueError("Cannot compare feature variants from different kernels")

    summaries: List[pd.DataFrame] = []
    disagreements: List[pd.DataFrame] = []

    for experiment in exact_result.predictions:
        _, summary, disagreement = compare_prediction_frames(
            exact_result.predictions[experiment],
            sampled_result.predictions[experiment],
            experiment,
            out_dir,
        )
        summaries.append(summary)
        disagreements.append(disagreement)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_disagreement = pd.concat(disagreements, ignore_index=True)
    all_summary.to_csv(
        out_dir / "all_experiments_exact_vs_sampled_summary.csv", index=False
    )
    all_disagreement.to_csv(
        out_dir / "all_experiments_gate_disagreement.csv", index=False
    )

    exact_master = exact_result.master_summary.copy()
    sampled_master = sampled_result.master_summary.copy()
    metrics = [
        "mae_log",
        "rmse_log",
        "r2_log",
        "spearman",
        "pearson_log",
        "mae_speedup",
        "rmse_speedup",
        "mape",
        "smape",
    ]

    exact_wide = exact_master[["kernel", "experiment", *metrics]].rename(
        columns={metric: f"exact_{metric}" for metric in metrics}
    )
    sampled_wide = sampled_master[["kernel", "experiment", *metrics]].rename(
        columns={metric: f"sampled_{metric}" for metric in metrics}
    )
    metric_comparison = pd.merge(
        exact_wide,
        sampled_wide,
        on=["kernel", "experiment"],
        validate="one_to_one",
    )
    for metric in metrics:
        metric_comparison[f"sampled_minus_exact_{metric}"] = (
            metric_comparison[f"sampled_{metric}"]
            - metric_comparison[f"exact_{metric}"]
        )
    metric_comparison.to_csv(
        out_dir / "exact_vs_sampled_metric_comparison.csv", index=False
    )
    return metric_comparison

def evaluate_existing_model_on_feature_view(
    pipe: Pipeline,
    test_df: pd.DataFrame,
    trained_variant: str,
    input_variant: str,
    dataset_name: str,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    pred_log = pipe.predict(test_df[canonical_r_cols()])
    metrics = evaluate_regression(
        np.log(test_df["speedup"].to_numpy(dtype=float)), pred_log
    )
    experiment = f"cross_train_{trained_variant}_input_{input_variant}_{dataset_name}"
    pred_df = prediction_dataframe(test_df, pred_log, input_variant, experiment)
    row = {
        "kernel": str(test_df["kernel"].iloc[0]),
        "trained_feature_variant": trained_variant,
        "input_feature_variant": input_variant,
        "dataset": dataset_name,
        "experiment": experiment,
        **metrics,
    }
    return row, pred_df

def run_cross_feature_diagnostics(
    exact_result: VariantResult,
    sampled_result: VariantResult,
    sparse_dual: pd.DataFrame,
    rmat_dual: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    if exact_result.kernel != sampled_result.kernel:
        raise ValueError("Cross-feature diagnostics require the same kernel")

    rows: List[Dict[str, object]] = []

    for dataset_name, dual_df in {"sparse": sparse_dual, "rmat": rmat_dual}.items():
        exact_view = make_feature_view(dual_df, "exact")
        sampled_view = make_feature_view(dual_df, "sampled")

        combinations = [
            (exact_result.model_all_aug, "exact", exact_view, "exact"),
            (exact_result.model_all_aug, "exact", sampled_view, "sampled"),
            (sampled_result.model_all_aug, "sampled", sampled_view, "sampled"),
            (sampled_result.model_all_aug, "sampled", exact_view, "exact"),
        ]

        for pipe, trained_variant, input_df, input_variant in combinations:
            row, pred_df = evaluate_existing_model_on_feature_view(
                pipe,
                input_df,
                trained_variant,
                input_variant,
                dataset_name,
            )
            rows.append(row)
            pred_df.to_csv(
                out_dir / f"{row['experiment']}_predictions.csv", index=False
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "cross_feature_compatibility_summary.csv", index=False)
    return summary


# =========================================================
# Multi-backend orchestration and diagnostics
# =========================================================


def resolve_dataset_path(
    data_dir: Path,
    override: Optional[Path],
    default_name: str,
) -> Path:
    """Resolve an optional CLI override or a file under --data-dir."""
    return (override if override is not None else data_dir / default_name).resolve()


def load_kernel_bundle(
    kernel: str,
    train_path: Path,
    sparse_path: Path,
    rmat_path: Path,
) -> KernelBundle:
    return KernelBundle(
        kernel=kernel,
        train=load_dual_feature_dataset(
            train_path, "train_augmented", "swap", kernel
        ),
        sparse=load_dual_feature_dataset(
            sparse_path, "test_sparse_original", "sparse", kernel
        ),
        rmat=load_dual_feature_dataset(
            rmat_path, "test_rmat_external", "rmat", kernel
        ),
    )


def validate_cross_kernel_feature_consistency(
    bundles: Mapping[str, KernelBundle],
    out_dir: Path,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """
    RD features and extraction times are backend-independent. When both MERBIT
    and CSR datasets are supplied, verify that corresponding rows contain the
    same exact/sample features and timing measurements. Speedup targets may differ.
    """
    if not {"merbit", "csr"}.issubset(bundles):
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    feature_and_time_cols = [
        *source_r_cols("exact"),
        "exact_time_ms",
        *source_r_cols("sampled"),
        "sampled_time_ms",
    ]

    for split_name in ["train", "sparse", "rmat"]:
        merbit_df = getattr(bundles["merbit"], split_name)
        csr_df = getattr(bundles["csr"], split_name)

        left = merbit_df[["graph_id", "base_graph", *feature_and_time_cols]].copy()
        right = csr_df[["graph_id", "base_graph", *feature_and_time_cols]].copy()
        merged = pd.merge(
            left,
            right,
            on=["graph_id", "base_graph"],
            how="outer",
            suffixes=("_merbit", "_csr"),
            indicator=True,
            validate="one_to_one",
        )

        unmatched = merged[merged["_merge"] != "both"]
        if len(unmatched) > 0:
            examples = unmatched[["graph_id", "_merge"]].head(10).to_dict("records")
            raise ValueError(
                f"[{split_name}] MERBIT/CSR graph IDs do not align. Examples: {examples}"
            )

        max_diff = 0.0
        per_column_max: Dict[str, float] = {}
        for column in feature_and_time_cols:
            a = merged[f"{column}_merbit"].to_numpy(dtype=float)
            b = merged[f"{column}_csr"].to_numpy(dtype=float)
            diff = np.abs(a - b)
            column_max = float(np.max(diff)) if len(diff) else 0.0
            per_column_max[column] = column_max
            max_diff = max(max_diff, column_max)

        rows.append(
            {
                "split": split_name,
                "rows": int(len(merged)),
                "max_abs_feature_or_time_difference": max_diff,
                **{f"max_abs_diff_{key}": value for key, value in per_column_max.items()},
            }
        )

        if max_diff > atol:
            raise ValueError(
                f"[{split_name}] backend-independent RD data differ between MERBIT "
                f"and CSR: max absolute difference={max_diff:.12g} > {atol:.12g}"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "cross_kernel_feature_consistency.csv", index=False)
    return summary


def common_suite_sparse_graphs(
    bundles: Mapping[str, KernelBundle],
) -> List[str]:
    graph_sets: List[set[str]] = []
    for bundle in bundles.values():
        graph_sets.append(set(bundle.train["base_graph"].astype(str)))
        graph_sets.append(set(bundle.sparse["base_graph"].astype(str)))

    common = set.intersection(*graph_sets) if graph_sets else set()
    if not common:
        raise ValueError("No common SuiteSparse graphs across selected datasets")

    for kernel, bundle in bundles.items():
        train_set = set(bundle.train["base_graph"].astype(str))
        sparse_set = set(bundle.sparse["base_graph"].astype(str))
        if train_set != common or sparse_set != common:
            raise ValueError(
                f"[{kernel}] graph sets differ from the common set: "
                f"train={len(train_set)}, sparse={len(sparse_set)}, common={len(common)}"
            )

    return sorted(common, key=natural_graph_key)


def compare_backend_prediction_frames(
    merbit_df: pd.DataFrame,
    csr_df: pd.DataFrame,
    feature_variant: str,
    experiment: str,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare targets and predictions for the same graphs under MERBIT and CSR."""
    keys = ["graph_id", "base_graph"]
    if "fold" in merbit_df.columns and "fold" in csr_df.columns:
        keys.append("fold")

    selected = [*keys, "speedup", "pred_speedup", "pred_log_speedup"]
    merbit_part = merbit_df[selected].rename(
        columns={
            "speedup": "true_speedup_merbit",
            "pred_speedup": "pred_speedup_merbit",
            "pred_log_speedup": "pred_log_speedup_merbit",
        }
    )
    csr_part = csr_df[selected].rename(
        columns={
            "speedup": "true_speedup_csr",
            "pred_speedup": "pred_speedup_csr",
            "pred_log_speedup": "pred_log_speedup_csr",
        }
    )
    paired = pd.merge(
        merbit_part,
        csr_part,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(merbit_df) or len(paired) != len(csr_df):
        raise RuntimeError(
            f"[{feature_variant}/{experiment}] MERBIT/CSR prediction alignment failed: "
            f"merbit={len(merbit_df)}, csr={len(csr_df)}, merged={len(paired)}"
        )

    paired["feature_variant"] = feature_variant
    paired["experiment"] = experiment
    paired["actual_beneficial_merbit"] = paired["true_speedup_merbit"] > 1.0
    paired["actual_beneficial_csr"] = paired["true_speedup_csr"] > 1.0
    paired["actual_backend_disagreement"] = (
        paired["actual_beneficial_merbit"] != paired["actual_beneficial_csr"]
    )
    for tau_h in sorted({pair[1] for pair in BAND_GATE_PAIRS}):
        paired[f"pred_select_merbit_tau_{tau_h:.2f}"] = (
            paired["pred_speedup_merbit"] >= tau_h
        )
        paired[f"pred_select_csr_tau_{tau_h:.2f}"] = (
            paired["pred_speedup_csr"] >= tau_h
        )

    paired.to_csv(
        out_dir / f"{feature_variant}_{experiment}_merbit_vs_csr_predictions.csv",
        index=False,
    )

    true_m = paired["true_speedup_merbit"].to_numpy(dtype=float)
    true_c = paired["true_speedup_csr"].to_numpy(dtype=float)
    pred_m = paired["pred_speedup_merbit"].to_numpy(dtype=float)
    pred_c = paired["pred_speedup_csr"].to_numpy(dtype=float)

    summary_row: Dict[str, object] = {
        "feature_variant": feature_variant,
        "experiment": experiment,
        "rows": int(len(paired)),
        "mean_true_speedup_merbit": float(np.mean(true_m)),
        "mean_true_speedup_csr": float(np.mean(true_c)),
        "median_true_speedup_merbit": float(np.median(true_m)),
        "median_true_speedup_csr": float(np.median(true_c)),
        "spearman_true_merbit_vs_csr": spearman_corr(true_m, true_c),
        "pearson_log_true_merbit_vs_csr": pearson_corr(np.log(true_m), np.log(true_c)),
        "spearman_pred_merbit_vs_csr": spearman_corr(pred_m, pred_c),
        "beneficial_count_merbit": int(np.sum(true_m > 1.0)),
        "beneficial_count_csr": int(np.sum(true_c > 1.0)),
        "actual_backend_disagreement_count": int(
            paired["actual_backend_disagreement"].sum()
        ),
        "actual_backend_disagreement_rate": float(
            paired["actual_backend_disagreement"].mean()
        ),
    }

    for tau_h in sorted({pair[1] for pair in BAND_GATE_PAIRS}):
        merbit_select = pred_m >= tau_h
        csr_select = pred_c >= tau_h
        summary_row[f"gate_disagreement_rate_tau_{tau_h:.2f}"] = float(
            np.mean(merbit_select != csr_select)
        )

    summary = pd.DataFrame([summary_row])
    summary.to_csv(
        out_dir / f"{feature_variant}_{experiment}_merbit_vs_csr_summary.csv",
        index=False,
    )
    return paired, summary


def run_backend_sensitivity_analysis(
    kernel_results: Mapping[str, KernelResult],
    out_dir: Path,
) -> pd.DataFrame:
    if not {"merbit", "csr"}.issubset(kernel_results):
        return pd.DataFrame()

    summaries: List[pd.DataFrame] = []
    for variant in FEATURE_VARIANTS:
        merbit_result = kernel_results["merbit"].variants[variant]
        csr_result = kernel_results["csr"].variants[variant]
        for experiment in merbit_result.predictions:
            _, summary = compare_backend_prediction_frames(
                merbit_result.predictions[experiment],
                csr_result.predictions[experiment],
                variant,
                experiment,
                out_dir,
            )
            summaries.append(summary)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(out_dir / "all_backend_sensitivity_summary.csv", index=False)
    return all_summary


def run_kernel_pipeline(
    bundle: KernelBundle,
    graph_folds: Sequence[GraphFold],
    ridge_alpha: float,
    expected_augmented_per_graph: Optional[int],
    root_out_dir: Path,
) -> KernelResult:
    kernel_dir = ensure_directory(root_out_dir / bundle.kernel)
    comparison_dir = ensure_directory(kernel_dir / "comparison")
    diagnostics_dir = ensure_directory(kernel_dir / "cross_feature_diagnostics")

    validate_group_structure(
        bundle.train,
        bundle.sparse,
        expected_augmented_per_graph,
        kernel_dir,
    )

    results: Dict[str, VariantResult] = {}
    for variant in FEATURE_VARIANTS:
        print(
            f"\n==================== {bundle.kernel.upper()} / "
            f"{variant.upper()} FEATURES ===================="
        )
        results[variant] = run_variant(
            bundle.kernel,
            variant,
            bundle.train,
            bundle.sparse,
            bundle.rmat,
            graph_folds,
            ridge_alpha,
            kernel_dir,
        )

    metric_comparison = compare_variants(
        results["exact"], results["sampled"], comparison_dir
    )
    cross_feature_summary = run_cross_feature_diagnostics(
        results["exact"],
        results["sampled"],
        bundle.sparse,
        bundle.rmat,
        diagnostics_dir,
    )

    all_master = pd.concat(
        [results[variant].master_summary for variant in FEATURE_VARIANTS],
        ignore_index=True,
    )
    all_master.to_csv(kernel_dir / "all_variants_master_summary.csv", index=False)

    all_coefficients = pd.concat(
        [results[variant].coefficient_table for variant in FEATURE_VARIANTS],
        ignore_index=True,
    )
    all_coefficients.to_csv(
        kernel_dir / "all_variants_deploy_model_parameters.csv", index=False
    )

    return KernelResult(
        kernel=bundle.kernel,
        variants=results,
        metric_comparison=metric_comparison,
        cross_feature_summary=cross_feature_summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact vs sampled RD predictors for MERBIT and CSR using "
            "shared graph folds."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the six default CSV files.",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=SUPPORTED_KERNELS,
        default=list(SUPPORTED_KERNELS),
        help="Backends to evaluate. Default: merbit csr.",
    )

    parser.add_argument("--train-merbit-csv", type=Path, default=None)
    parser.add_argument("--sparse-test-merbit-csv", type=Path, default=None)
    parser.add_argument("--rmat-test-merbit-csv", type=Path, default=None)
    parser.add_argument("--train-csr-csv", type=Path, default=None)
    parser.add_argument("--sparse-test-csr-csv", type=Path, default=None)
    parser.add_argument("--rmat-test-csr-csv", type=Path, default=None)

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("predictor_eval_exact_sampled_dual_backend_outputs"),
        help="Output directory.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument(
        "--expected-augmented-per-graph",
        type=int,
        default=9,
        help="Set to 0 to disable the group-size assertion.",
    )
    parser.add_argument(
        "--cross-kernel-feature-tol",
        type=float,
        default=1e-12,
        help=(
            "Maximum allowed difference between backend-independent RD fields "
            "in corresponding MERBIT and CSR files."
        ),
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.ridge_alpha < 0.0:
        raise ValueError("ridge-alpha must be non-negative")
    if args.cv_folds < 2:
        raise ValueError("cv-folds must be at least 2")
    if args.cross_kernel_feature_tol < 0.0:
        raise ValueError("cross-kernel-feature-tol must be non-negative")

    selected_kernels = list(dict.fromkeys(args.kernels))
    data_dir = args.data_dir.resolve()
    out_dir = ensure_directory(args.out_dir.resolve())
    feature_dir = ensure_directory(out_dir / "feature_analysis")
    backend_dir = ensure_directory(out_dir / "backend_sensitivity")

    override_map: Dict[str, Dict[str, Optional[Path]]] = {
        "merbit": {
            "train": args.train_merbit_csv,
            "sparse": args.sparse_test_merbit_csv,
            "rmat": args.rmat_test_merbit_csv,
        },
        "csr": {
            "train": args.train_csr_csv,
            "sparse": args.sparse_test_csr_csv,
            "rmat": args.rmat_test_csr_csv,
        },
    }

    resolved_paths: Dict[str, Dict[str, Path]] = {}
    bundles: Dict[str, KernelBundle] = {}
    for kernel in selected_kernels:
        resolved_paths[kernel] = {}
        for split_name, default_name in DEFAULT_DATA_FILES[kernel].items():
            resolved_paths[kernel][split_name] = resolve_dataset_path(
                data_dir,
                override_map[kernel][split_name],
                default_name,
            )

        bundles[kernel] = load_kernel_bundle(
            kernel,
            resolved_paths[kernel]["train"],
            resolved_paths[kernel]["sparse"],
            resolved_paths[kernel]["rmat"],
        )

    validate_cross_kernel_feature_consistency(
        bundles,
        out_dir,
        atol=args.cross_kernel_feature_tol,
    )

    all_dataset_frames: List[pd.DataFrame] = []
    for bundle in bundles.values():
        all_dataset_frames.extend([bundle.train, bundle.sparse, bundle.rmat])

    all_dataset_summaries = pd.concat(
        [dataset_summary(frame) for frame in all_dataset_frames],
        ignore_index=True,
    )
    all_dataset_summaries.to_csv(
        out_dir / "all_kernels_dataset_summaries.csv", index=False
    )

    # Exact/sample RD features are backend-independent. After consistency
    # validation, analyze them once using the first selected backend.
    reference_bundle = bundles[selected_kernels[0]]
    feature_summaries: List[pd.DataFrame] = []
    feature_per_bins: List[pd.DataFrame] = []
    for frame in [reference_bundle.train, reference_bundle.sparse, reference_bundle.rmat]:
        analysis = analyze_feature_approximation(frame, feature_dir)
        feature_summaries.append(analysis["summary"])
        feature_per_bins.append(analysis["per_bin"])
    pd.concat(feature_summaries, ignore_index=True).to_csv(
        feature_dir / "all_datasets_feature_approximation_summary.csv", index=False
    )
    pd.concat(feature_per_bins, ignore_index=True).to_csv(
        feature_dir / "all_datasets_feature_approximation_per_bin.csv", index=False
    )

    shared_graphs = common_suite_sparse_graphs(bundles)
    graph_folds = build_graph_folds(shared_graphs, args.cv_folds)
    save_graph_folds(graph_folds, out_dir / "shared_suite_sparse_graph_folds.csv")

    kernel_results: Dict[str, KernelResult] = {}
    for kernel in selected_kernels:
        kernel_results[kernel] = run_kernel_pipeline(
            bundles[kernel],
            graph_folds,
            args.ridge_alpha,
            args.expected_augmented_per_graph or None,
            out_dir,
        )

    backend_summary = run_backend_sensitivity_analysis(kernel_results, backend_dir)

    all_master = pd.concat(
        [
            result.variants[variant].master_summary
            for result in kernel_results.values()
            for variant in FEATURE_VARIANTS
        ],
        ignore_index=True,
    )
    all_master.to_csv(out_dir / "all_kernels_master_summary.csv", index=False)

    all_metric_comparison = pd.concat(
        [result.metric_comparison for result in kernel_results.values()],
        ignore_index=True,
    )
    all_metric_comparison.to_csv(
        out_dir / "all_kernels_exact_vs_sampled_metric_comparison.csv",
        index=False,
    )

    all_cross_feature = pd.concat(
        [result.cross_feature_summary for result in kernel_results.values()],
        ignore_index=True,
    )
    all_cross_feature.to_csv(
        out_dir / "all_kernels_cross_feature_compatibility_summary.csv",
        index=False,
    )

    all_coefficients = pd.concat(
        [
            result.variants[variant].coefficient_table
            for result in kernel_results.values()
            for variant in FEATURE_VARIANTS
        ],
        ignore_index=True,
    )
    all_coefficients.to_csv(
        out_dir / "all_kernels_deploy_model_parameters.csv", index=False
    )

    run_metadata = {
        "data_dir": str(data_dir),
        "selected_kernels": selected_kernels,
        "input_files": {
            kernel: {
                split_name: str(path)
                for split_name, path in paths.items()
            }
            for kernel, paths in resolved_paths.items()
        },
        "out_dir": str(out_dir),
        "ridge_alpha": float(args.ridge_alpha),
        "cv_folds": int(args.cv_folds),
        "feature_variants": list(FEATURE_VARIANTS),
        "feature_count": FEATURE_COUNT,
        "band_gate_pairs": BAND_GATE_PAIRS,
        "expected_augmented_per_graph": int(args.expected_augmented_per_graph),
        "cross_kernel_feature_tolerance": float(args.cross_kernel_feature_tol),
        "random_seed_reserved": RANDOM_SEED,
    }
    write_json(out_dir / "run_metadata.json", run_metadata)

    print("\n==================== ALL-KERNEL MASTER SUMMARY ====================")
    print(all_master.to_string(index=False))
    print("\n==================== EXACT VS SAMPLED DELTAS ====================")
    print(all_metric_comparison.to_string(index=False))
    print("\n==================== CROSS-FEATURE DIAGNOSTIC ====================")
    print(all_cross_feature.to_string(index=False))
    if len(backend_summary) > 0:
        print("\n==================== MERBIT VS CSR SENSITIVITY ====================")
        print(backend_summary.to_string(index=False))

    print(f"\nDone. Outputs written to: {out_dir}")
    for kernel in selected_kernels:
        print(
            f"Primary {kernel.upper()} deployment header: "
            f"{out_dir / kernel / 'sampled' / f'rd_sampled_{kernel}_model.hpp'}"
        )


if __name__ == "__main__":
    main()
