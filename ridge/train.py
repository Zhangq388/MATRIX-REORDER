"""
Evaluate the RD-based SpMV headroom predictor with two target kernels:
  1) MERBIT SpMV speedup targets
  2) CSR SpMV speedup targets

Input CSVs expected by default:
  - train_merbit.csv            augmented local-swap SuiteSparse samples
  - train_csr.csv               augmented local-swap SuiteSparse samples
  - test_sparse_merbit.csv      original SuiteSparse samples
  - test_sparse_csr.csv         original SuiteSparse samples

Each CSV should contain:
  graph_id, r1...r10, and either speedup or time.

Experiments:
  A) GroupKFold on augmented samples only.
  B) Train on all augmented samples and test on original samples from the same graph set.
     This reproduces the old S2O setting, but it should be described as same-graph testing.
  C) Leakage-free held-out graph test. For each fold, the model is trained only on
     augmented samples from training graphs and tested on original samples from unseen graphs.
     This is the recommended experiment for claiming generalization across real-world graphs.

Outputs are written to ./predictor_eval_merbit_csr_outputs by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# Config
# =========================================================

RIDGE_ALPHA = 1.0
EPS = 1e-12
CV_FOLDS = 5

# 三值门控阈值扫描。
BAND_GATE_PAIRS: List[Tuple[float, float]] = [
    (0.99, 1.01),
    (0.98, 1.02),
    (0.97, 1.03),
    (0.96, 1.04),
    (0.95, 1.05),
    (0.94, 1.06),
    (0.93, 1.07),
]

DEFAULT_DATASETS = {
    "merbit": {
        "train_csv": "train_merbit.csv",
        "test_csv": "test_sparse_merbit.csv",
    },
    "csr": {
        "train_csv": "train_csr.csv",
        "test_csv": "test_sparse_csr.csv",
    },
}


# =========================================================
# Utilities
# =========================================================


def spearman_corr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan

    y_true_rank = pd.Series(y_true).rank(method="average").to_numpy()
    y_pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()

    a = y_true_rank - y_true_rank.mean()
    b = y_pred_rank - y_pred_rank.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den < 1e-15:
        return np.nan
    return float(np.sum(a * b) / den)



def mape(y_true_speedup: Sequence[float], y_pred_speedup: Sequence[float], eps: float = EPS) -> float:
    y_true_speedup = np.asarray(y_true_speedup, dtype=float)
    y_pred_speedup = np.asarray(y_pred_speedup, dtype=float)
    denom = np.maximum(np.abs(y_true_speedup), eps)
    return float(np.mean(np.abs(y_pred_speedup - y_true_speedup) / denom))



def _is_number_like(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False



def _is_int_like(x: str) -> bool:
    return x.isdigit() or (x.startswith("-") and x[1:].isdigit())



def extract_base_graph_id_from_swap(graph_id: object) -> str:
    """
    Extract the original graph identity from a local-swap sample ID.

    Supported examples:
      1_0.1_64_256                 -> 1
      belgium_osm_0.1_64_256       -> belgium_osm

    The old implementation used split("_")[0], which is unsafe for matrix names
    that already contain underscores. This version removes the trailing local-swap
    parameters when they are detected.
    """
    s = str(graph_id).strip()
    parts = s.split("_")

    # Typical local-swap suffix: <base>_<ratio/window/etc>_<param2>_<param3>
    # Example: graph_0.1_64_256. Keep this conservative to avoid damaging names.
    if len(parts) >= 4 and _is_number_like(parts[-3]) and _is_int_like(parts[-2]) and _is_int_like(parts[-1]):
        return "_".join(parts[:-3])

    # Fallback for old numeric IDs such as 1_0.1_64_256 if suffix format changes.
    if len(parts) > 1 and _is_number_like(parts[1]):
        return parts[0]

    return s



def extract_base_graph_id_generic(graph_id: object) -> str:
    """For original SuiteSparse IDs, keep the whole graph_id."""
    return str(graph_id).strip()



def detect_r_cols(df: pd.DataFrame) -> List[str]:
    r_cols = [c for c in df.columns if c.startswith("r") and c[1:].isdigit()]
    return sorted(r_cols, key=lambda x: int(x[1:]))



def choose_target_column(df: pd.DataFrame, dataset_name: str) -> str:
    """
    The old script used a column named `time` as the speedup target.
    The new MERBIT / CSR CSVs use `speedup`. Support both formats.
    """
    if "speedup" in df.columns:
        return "speedup"
    if "time" in df.columns:
        return "time"
    raise ValueError(
        f"[{dataset_name}] CSV must contain either `speedup` or `time`. "
        f"Actual columns: {list(df.columns)}"
    )


# =========================================================
# Loading / Cleaning
# =========================================================


def load_and_clean(csv_path: Path, dataset_name: str, base_graph_mode: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    base_graph_mode:
      - swap:    graph_id=<base>_<swap_param1>_<swap_param2>_<swap_param3> -> base_graph=<base>
      - generic: graph_id itself is the base graph ID
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    r_cols = detect_r_cols(df)
    if not r_cols:
        raise ValueError(f"[{dataset_name}] No RD feature columns r1...rK found.")

    target_col = choose_target_column(df, dataset_name)
    required_cols = ["graph_id", target_col] + r_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{dataset_name}] Missing columns: {missing}. Actual columns: {list(df.columns)}")

    # If a method column is present, keep origin rows to match the old script behavior.
    if "method" in df.columns:
        before = len(df)
        df = df[df["method"].astype(str).str.strip().str.lower() == "origin"].copy()
        print(f"[{dataset_name}] filter method == origin: {before} -> {len(df)}")

    for c in r_cols + [target_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required_cols).copy()
    df = df[df[target_col] > 0].copy()

    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    if base_graph_mode == "swap":
        df["base_graph"] = df["graph_id"].apply(extract_base_graph_id_from_swap)
    elif base_graph_mode == "generic":
        df["base_graph"] = df["graph_id"].apply(extract_base_graph_id_generic)
    else:
        raise ValueError(f"Unknown base_graph_mode: {base_graph_mode}")

    # Normalize the target column name for downstream code.
    df["speedup"] = df[target_col].astype(float)

    rsum = df[r_cols].sum(axis=1)
    max_rsum_dev = float(np.max(np.abs(rsum - 1.0))) if len(df) else np.nan

    print(f"\n[{dataset_name}]")
    print(f"  source: {csv_path.name}")
    print(f"  target column: {target_col} -> speedup")
    print(f"  rows: {len(df)}")
    print(f"  unique graph_id: {df['graph_id'].nunique()}")
    print(f"  unique base_graph: {df['base_graph'].nunique()}")
    print(f"  r columns: {r_cols}")
    print(f"  max |sum(r)-1| = {max_rsum_dev:.6g}")
    print(df["speedup"].describe().to_string())

    return df, r_cols



def dataset_stats(df: pd.DataFrame, dataset_name: str, kernel: str) -> Dict[str, object]:
    y = df["speedup"].to_numpy(dtype=float)
    y_pos = y > 1.0
    return {
        "kernel": kernel,
        "dataset": dataset_name,
        "rows": int(len(df)),
        "graph_ids": int(df["graph_id"].nunique()),
        "base_graphs": int(df["base_graph"].nunique()),
        "mean_speedup": float(np.mean(y)),
        "median_speedup": float(np.median(y)),
        "min_speedup": float(np.min(y)),
        "max_speedup": float(np.max(y)),
        "num_pos_gt1": int(np.sum(y_pos)),
        "num_neg_le1": int(len(y_pos) - np.sum(y_pos)),
        "pos_rate_gt1": float(np.mean(y_pos)),
    }



def check_group_counts(df: pd.DataFrame, dataset_name: str, out_dir: Path, expected_group_size: Optional[int] = None) -> pd.DataFrame:
    cnt = df.groupby("base_graph").size().sort_values()
    out = pd.DataFrame({
        "dataset": dataset_name,
        "base_graph": cnt.index.astype(str),
        "num_samples": cnt.values,
    })
    out.to_csv(out_dir / f"{dataset_name}_group_counts.csv", index=False)

    print(f"\n[{dataset_name}] group size statistics")
    print(cnt.describe().to_string())
    if expected_group_size is not None:
        bad = cnt[cnt != expected_group_size]
        print(f"[{dataset_name}] groups not equal to {expected_group_size}: {len(bad)}")
        if len(bad) > 0:
            print(bad.to_string())
    return out



def check_train_test_graph_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, kernel: str, out_dir: Path) -> pd.DataFrame:
    train_graphs = set(train_df["base_graph"].astype(str))
    test_graphs = set(test_df["base_graph"].astype(str))
    common = sorted(train_graphs & test_graphs)
    train_only = sorted(train_graphs - test_graphs)
    test_only = sorted(test_graphs - train_graphs)

    rows = []
    for g in common:
        rows.append({"kernel": kernel, "base_graph": g, "status": "common"})
    for g in train_only:
        rows.append({"kernel": kernel, "base_graph": g, "status": "train_only"})
    for g in test_only:
        rows.append({"kernel": kernel, "base_graph": g, "status": "test_only"})

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"{kernel}_train_test_base_graph_overlap.csv", index=False)

    print(f"\n[{kernel}] train/test base_graph overlap")
    print(f"  common graphs: {len(common)}")
    print(f"  train-only graphs: {len(train_only)}")
    print(f"  test-only graphs: {len(test_only)}")

    if len(common) == 0:
        raise ValueError(
            f"[{kernel}] No common base_graph IDs between augmented train CSV and original test CSV. "
            "Please check graph_id naming or the base_graph extraction rule."
        )
    return out


# =========================================================
# Model
# =========================================================


def build_preprocess(r_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                r_cols,
            ),
        ]
    )



def make_pipe(r_cols: List[str]) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocess(r_cols)),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])


# =========================================================
# Evaluation
# =========================================================


def eval_regression(y_true_log: Sequence[float], y_pred_log: Sequence[float]) -> Dict[str, float]:
    y_true_s = np.exp(np.asarray(y_true_log, dtype=float))
    y_pred_s = np.exp(np.asarray(y_pred_log, dtype=float))
    return {
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "spearman": float(spearman_corr(y_true_log, y_pred_log)),
        "mae_speedup": float(mean_absolute_error(y_true_s, y_pred_s)),
        "mape": float(mape(y_true_s, y_pred_s, eps=EPS)),
    }



def run_group_cv_once(train_df: pd.DataFrame, r_cols: List[str], cv_folds: int, save_prefix: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = train_df[r_cols].copy().reset_index(drop=True)
    y_log = np.log(train_df["speedup"].to_numpy(dtype=float))
    groups = train_df["base_graph"].reset_index(drop=True)

    n_groups = groups.nunique()
    if cv_folds > n_groups:
        raise ValueError(f"cv_folds={cv_folds} > number of groups={n_groups}")

    gkf = GroupKFold(n_splits=cv_folds)
    oof_pred_log = np.zeros_like(y_log)
    rows: List[Dict[str, object]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(X, y_log, groups=groups), start=1):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y_log[tr_idx]
        y_va = y_log[va_idx]

        tr_groups = set(groups.iloc[tr_idx].tolist())
        va_groups = set(groups.iloc[va_idx].tolist())
        overlap = tr_groups & va_groups
        if overlap:
            raise ValueError(f"Fold {fold_id} has group leakage: {overlap}")

        pipe = make_pipe(r_cols)
        pipe.fit(X_tr, y_tr)
        pred_log = pipe.predict(X_va)
        oof_pred_log[va_idx] = pred_log
        metrics = eval_regression(y_va, pred_log)

        rows.append({
            "fold": fold_id,
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
            "train_base_graphs": int(len(tr_groups)),
            "valid_base_graphs": int(len(va_groups)),
            **metrics,
        })

    folds_df = pd.DataFrame(rows)
    oof_metrics = eval_regression(y_log, oof_pred_log)
    summary_df = pd.DataFrame([{
        "cv_type": "GroupKFold(base_graph) on augmented samples",
        "cv_folds": int(cv_folds),
        "mae_log_mean": float(folds_df["mae_log"].mean()),
        "mae_log_std": float(folds_df["mae_log"].std(ddof=1)),
        "spearman_mean": float(folds_df["spearman"].mean()),
        "spearman_std": float(folds_df["spearman"].std(ddof=1)),
        "mae_speedup_mean": float(folds_df["mae_speedup"].mean()),
        "mae_speedup_std": float(folds_df["mae_speedup"].std(ddof=1)),
        "mape_mean": float(folds_df["mape"].mean()),
        "mape_std": float(folds_df["mape"].std(ddof=1)),
        "oof_mae_log": oof_metrics["mae_log"],
        "oof_spearman": oof_metrics["spearman"],
        "oof_mae_speedup": oof_metrics["mae_speedup"],
        "oof_mape": oof_metrics["mape"],
    }])

    oof_df = train_df[["graph_id", "base_graph", "speedup"] + r_cols].copy().reset_index(drop=True)
    oof_df["true_log_speedup"] = y_log
    oof_df["pred_log_speedup"] = oof_pred_log
    oof_df["pred_speedup"] = np.exp(oof_pred_log)
    oof_df["abs_err_speedup"] = np.abs(oof_df["pred_speedup"] - oof_df["speedup"])
    oof_df["ape"] = oof_df["abs_err_speedup"] / np.maximum(np.abs(oof_df["speedup"]), EPS)

    folds_df.to_csv(f"{save_prefix}_folds.csv", index=False)
    summary_df.to_csv(f"{save_prefix}_summary.csv", index=False)
    oof_df.to_csv(f"{save_prefix}_oof_predictions.csv", index=False)

    return folds_df, summary_df, oof_df



def fit_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, r_cols: List[str], exp_name: str, out_dir: Path) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = train_df[r_cols].copy()
    X_test = test_df[r_cols].copy()
    y_train_log = np.log(train_df["speedup"].to_numpy(dtype=float))
    y_test_log = np.log(test_df["speedup"].to_numpy(dtype=float))

    pipe = make_pipe(r_cols)
    pipe.fit(X_train, y_train_log)
    y_pred_log = pipe.predict(X_test)
    metrics = eval_regression(y_test_log, y_pred_log)

    pred_df = test_df[["graph_id", "base_graph", "speedup"] + r_cols].copy()
    pred_df["true_log_speedup"] = y_test_log
    pred_df["pred_log_speedup"] = y_pred_log
    pred_df["pred_speedup"] = np.exp(y_pred_log)
    pred_df["abs_err_speedup"] = np.abs(pred_df["pred_speedup"] - pred_df["speedup"])
    pred_df["ape"] = pred_df["abs_err_speedup"] / np.maximum(np.abs(pred_df["speedup"]), EPS)
    pred_df.to_csv(out_dir / f"{exp_name}_predictions.csv", index=False)

    graph_rows: List[Dict[str, object]] = []
    for g, sub in pred_df.groupby("base_graph"):
        graph_rows.append({
            "base_graph": g,
            "n_samples": int(len(sub)),
            "mae_log": float(mean_absolute_error(sub["true_log_speedup"], sub["pred_log_speedup"])),
            "spearman_log": float(spearman_corr(sub["true_log_speedup"], sub["pred_log_speedup"])),
            "mae_speedup": float(mean_absolute_error(sub["speedup"], sub["pred_speedup"])),
            "mape": float(mape(sub["speedup"], sub["pred_speedup"], eps=EPS)),
            "true_speedup_mean": float(sub["speedup"].mean()),
            "pred_speedup_mean": float(sub["pred_speedup"].mean()),
        })
    per_graph_df = pd.DataFrame(graph_rows).sort_values("mape", ascending=False)
    per_graph_df.to_csv(out_dir / f"{exp_name}_per_graph.csv", index=False)

    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)
    coef_df.to_csv(out_dir / f"{exp_name}_coefficients.csv", index=False)

    return pipe, metrics, pred_df, per_graph_df, coef_df



def run_band_gate_scan(pred_df: pd.DataFrame, tau_pairs: Iterable[Tuple[float, float]], save_csv_path: Path) -> pd.DataFrame:
    """
    三值门控:
      relabel    if pred >= tau_h
      skip       if pred <= tau_l
      uncertain  otherwise

    正类: true speedup > 1.
    Coverage / Accuracy / Precision / Recall are computed on definite decisions only.
    """
    rows: List[Dict[str, object]] = []
    y_true = pred_df["speedup"].to_numpy(dtype=float)
    y_pred = pred_df["pred_speedup"].to_numpy(dtype=float)
    y_true_pos = y_true > 1.0

    for tau_l, tau_h in tau_pairs:
        decision = np.full(len(pred_df), "uncertain", dtype=object)
        decision[y_pred >= tau_h] = "relabel"
        decision[y_pred <= tau_l] = "skip"

        decided_mask = decision != "uncertain"
        n_total = len(pred_df)
        n_decided = int(np.sum(decided_mask))
        coverage = n_decided / n_total if n_total else np.nan

        if n_decided == 0:
            rows.append({
                "tau_l": tau_l,
                "tau_h": tau_h,
                "coverage": coverage,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "num_decided": 0,
                "num_total": n_total,
            })
            continue

        d_true = y_true_pos[decided_mask]
        d_pred_pos = decision[decided_mask] == "relabel"
        tp = int(np.sum((d_pred_pos == 1) & (d_true == 1)))
        tn = int(np.sum((d_pred_pos == 0) & (d_true == 0)))
        fp = int(np.sum((d_pred_pos == 1) & (d_true == 0)))
        fn = int(np.sum((d_pred_pos == 0) & (d_true == 1)))
        denom_acc = tp + tn + fp + fn

        rows.append({
            "tau_l": tau_l,
            "tau_h": tau_h,
            "coverage": coverage,
            "accuracy": (tp + tn) / denom_acc if denom_acc else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "num_decided": n_decided,
            "num_total": n_total,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(save_csv_path, index=False)
    return out_df



def run_conservative_gate(pred_df: pd.DataFrame, tau_h_values: Iterable[float], save_csv_path: Path) -> pd.DataFrame:
    """
    Deployment-style gate:
      relabel only if pred_speedup >= tau_h; otherwise keep original labeling.

    effective_speedup = true_speedup for selected samples, 1.0 otherwise.
    This estimates the average speedup achieved by the predictor gate itself,
    not just the classification quality.
    """
    rows: List[Dict[str, object]] = []
    y_true = pred_df["speedup"].to_numpy(dtype=float)
    y_pred = pred_df["pred_speedup"].to_numpy(dtype=float)
    true_pos = y_true > 1.0

    for tau_h in tau_h_values:
        relabel = y_pred >= tau_h
        effective = np.where(relabel, y_true, 1.0)
        selected = int(np.sum(relabel))
        bad_selected = int(np.sum(relabel & (~true_pos)))
        missed_good = int(np.sum((~relabel) & true_pos))
        rows.append({
            "tau_h": tau_h,
            "selected_relabel": selected,
            "selection_rate": selected / len(pred_df) if len(pred_df) else np.nan,
            "bad_selected_speedup_le1": bad_selected,
            "missed_good_speedup_gt1": missed_good,
            "mean_effective_speedup": float(np.mean(effective)),
            "median_effective_speedup": float(np.median(effective)),
            "oracle_relabel_all_mean_speedup": float(np.mean(y_true)),
            "oracle_relabel_only_true_positive_mean_effective_speedup": float(np.mean(np.where(true_pos, y_true, 1.0))),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(save_csv_path, index=False)
    return out_df



def run_heldout_original_graph_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    r_cols: List[str],
    kernel: str,
    cv_folds: int,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leakage-free real-world generalization test.

    Split by base_graph. For each fold:
      - training set: augmented local-swap samples from training base graphs only;
      - test set: original SuiteSparse samples from held-out base graphs only.

    This directly addresses the S2O leakage/generalization concern: no augmented
    version of a held-out test graph appears in the training set.
    """
    exp_name = f"{kernel}_expC_heldoutGraphs_trainAug_testOriginal"

    train_graphs = set(train_df["base_graph"].astype(str))
    test_graphs = set(test_df["base_graph"].astype(str))
    common_graphs = sorted(train_graphs & test_graphs)

    if len(common_graphs) < cv_folds:
        raise ValueError(
            f"[{kernel}] Need at least {cv_folds} common base graphs for held-out graph CV, "
            f"but only found {len(common_graphs)}."
        )

    # Keep only graphs that exist in both augmented-train and original-test CSVs.
    train_common = train_df[train_df["base_graph"].isin(common_graphs)].copy()
    test_common = test_df[test_df["base_graph"].isin(common_graphs)].copy()

    split_df = pd.DataFrame({"base_graph": common_graphs})
    dummy_x = np.zeros((len(split_df), 1))
    dummy_y = np.zeros(len(split_df))
    groups = split_df["base_graph"].to_numpy()

    gkf = GroupKFold(n_splits=cv_folds)
    fold_rows: List[Dict[str, object]] = []
    pred_parts: List[pd.DataFrame] = []
    coef_parts: List[pd.DataFrame] = []
    split_rows: List[Dict[str, object]] = []

    for fold_id, (tr_g_idx, te_g_idx) in enumerate(gkf.split(dummy_x, dummy_y, groups=groups), start=1):
        fold_train_graphs = set(split_df.iloc[tr_g_idx]["base_graph"].astype(str))
        fold_test_graphs = set(split_df.iloc[te_g_idx]["base_graph"].astype(str))
        overlap = fold_train_graphs & fold_test_graphs
        if overlap:
            raise ValueError(f"[{kernel}] Held-out fold {fold_id} has group leakage: {sorted(overlap)[:10]}")

        fold_train_df = train_common[train_common["base_graph"].isin(fold_train_graphs)].copy()
        fold_test_df = test_common[test_common["base_graph"].isin(fold_test_graphs)].copy()

        if len(fold_train_df) == 0 or len(fold_test_df) == 0:
            raise ValueError(f"[{kernel}] Fold {fold_id} has empty train or test data.")

        fold_exp_name = f"{exp_name}_fold{fold_id}"
        _, metrics, fold_pred_df, _, fold_coef_df = fit_and_eval(
            fold_train_df,
            fold_test_df,
            r_cols,
            exp_name=fold_exp_name,
            out_dir=out_dir,
        )

        fold_pred_df = fold_pred_df.copy()
        fold_pred_df["fold"] = fold_id
        pred_parts.append(fold_pred_df)

        fold_coef_df = fold_coef_df.copy()
        fold_coef_df["fold"] = fold_id
        coef_parts.append(fold_coef_df)

        fold_rows.append({
            "kernel": kernel,
            "experiment": exp_name,
            "fold": fold_id,
            "train_aug_rows": int(len(fold_train_df)),
            "test_original_rows": int(len(fold_test_df)),
            "train_base_graphs": int(len(fold_train_graphs)),
            "test_base_graphs": int(len(fold_test_graphs)),
            "mae_log": metrics["mae_log"],
            "spearman": metrics["spearman"],
            "mae_speedup": metrics["mae_speedup"],
            "mape": metrics["mape"],
        })

        for g in sorted(fold_train_graphs):
            split_rows.append({"kernel": kernel, "experiment": exp_name, "fold": fold_id, "base_graph": g, "role": "train"})
        for g in sorted(fold_test_graphs):
            split_rows.append({"kernel": kernel, "experiment": exp_name, "fold": fold_id, "base_graph": g, "role": "test"})

    folds_df = pd.DataFrame(fold_rows)
    pred_oof_df = pd.concat(pred_parts, ignore_index=True)
    coef_all_df = pd.concat(coef_parts, ignore_index=True)
    split_all_df = pd.DataFrame(split_rows)

    # Overall out-of-fold metrics on original held-out graphs.
    overall = eval_regression(pred_oof_df["true_log_speedup"], pred_oof_df["pred_log_speedup"])
    summary_df = pd.DataFrame([{
        "kernel": kernel,
        "experiment": exp_name,
        "split_type": "Leakage-free GroupKFold by original base_graph",
        "cv_folds": int(cv_folds),
        "train_source": "augmented local-swap samples from training graphs only",
        "test_source": "original samples from held-out graphs only",
        "common_base_graphs_used": int(len(common_graphs)),
        "oof_test_rows": int(len(pred_oof_df)),
        "fold_mae_log_mean": float(folds_df["mae_log"].mean()),
        "fold_mae_log_std": float(folds_df["mae_log"].std(ddof=1)),
        "fold_spearman_mean": float(folds_df["spearman"].mean()),
        "fold_spearman_std": float(folds_df["spearman"].std(ddof=1)),
        "fold_mae_speedup_mean": float(folds_df["mae_speedup"].mean()),
        "fold_mae_speedup_std": float(folds_df["mae_speedup"].std(ddof=1)),
        "fold_mape_mean": float(folds_df["mape"].mean()),
        "fold_mape_std": float(folds_df["mape"].std(ddof=1)),
        "oof_mae_log": overall["mae_log"],
        "oof_spearman": overall["spearman"],
        "oof_mae_speedup": overall["mae_speedup"],
        "oof_mape": overall["mape"],
    }])

    folds_df.to_csv(out_dir / f"{exp_name}_folds.csv", index=False)
    summary_df.to_csv(out_dir / f"{exp_name}_summary.csv", index=False)
    pred_oof_df.to_csv(out_dir / f"{exp_name}_oof_predictions.csv", index=False)
    coef_all_df.to_csv(out_dir / f"{exp_name}_coefficients_by_fold.csv", index=False)
    split_all_df.to_csv(out_dir / f"{exp_name}_graph_splits.csv", index=False)

    band_gate_df = run_band_gate_scan(
        pred_oof_df,
        BAND_GATE_PAIRS,
        save_csv_path=out_dir / f"{exp_name}_band_gate.csv",
    )
    conservative_gate_df = run_conservative_gate(
        pred_oof_df,
        tau_h_values=[pair[1] for pair in BAND_GATE_PAIRS],
        save_csv_path=out_dir / f"{exp_name}_conservative_gate.csv",
    )

    return summary_df, folds_df, pred_oof_df, band_gate_df, conservative_gate_df



def run_one_kernel(kernel: str, train_csv: Path, test_csv: Path, out_dir: Path) -> Dict[str, pd.DataFrame]:
    print(f"\n==================== {kernel.upper()} ====================")

    train_df, train_r_cols = load_and_clean(train_csv, f"{kernel}_train", base_graph_mode="swap")
    test_df, test_r_cols = load_and_clean(test_csv, f"{kernel}_test_sparse", base_graph_mode="generic")
    if train_r_cols != test_r_cols:
        raise ValueError(f"[{kernel}] r columns do not match: train={train_r_cols}, test={test_r_cols}")
    r_cols = train_r_cols

    stats_df = pd.DataFrame([
        dataset_stats(train_df, f"{kernel}_train", kernel),
        dataset_stats(test_df, f"{kernel}_test_sparse", kernel),
    ])
    stats_df.to_csv(out_dir / f"{kernel}_dataset_stats.csv", index=False)

    check_group_counts(train_df, f"{kernel}_train", out_dir, expected_group_size=9)
    check_group_counts(test_df, f"{kernel}_test_sparse", out_dir, expected_group_size=1)
    check_train_test_graph_overlap(train_df, test_df, kernel, out_dir)

    # Experiment A: 5-fold GroupKFold on the augmented samples.
    cv_prefix = out_dir / f"{kernel}_expA_5fold_groupkfold_on_trainAug"
    _, cv_summary_df, _ = run_group_cv_once(train_df, r_cols, cv_folds=CV_FOLDS, save_prefix=cv_prefix)

    # Experiment B: old S2O style. This is useful as an auxiliary same-graph test,
    # but it should not be used as the main evidence for unseen-graph generalization.
    exp_b_name = f"{kernel}_expB_trainAllAug_testOriginalSameGraphs"
    _, test_metrics, pred_b_df, _, coef_b_df = fit_and_eval(train_df, test_df, r_cols, exp_name=exp_b_name, out_dir=out_dir)
    exp_b_summary_df = pd.DataFrame([{
        "kernel": kernel,
        "experiment": exp_b_name,
        "split_type": "Same graph set: augmented versions of test graphs appear in training",
        "leakage_risk": "graph_identity_overlap",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_base_graphs": int(train_df["base_graph"].nunique()),
        "test_base_graphs": int(test_df["base_graph"].nunique()),
        "test_mae_log": test_metrics["mae_log"],
        "test_spearman": test_metrics["spearman"],
        "test_mae_speedup": test_metrics["mae_speedup"],
        "test_mape": test_metrics["mape"],
    }])
    exp_b_summary_df.to_csv(out_dir / f"{exp_b_name}_summary.csv", index=False)

    exp_b_band_gate_df = run_band_gate_scan(
        pred_b_df,
        BAND_GATE_PAIRS,
        save_csv_path=out_dir / f"{exp_b_name}_band_gate.csv",
    )
    exp_b_conservative_gate_df = run_conservative_gate(
        pred_b_df,
        tau_h_values=[pair[1] for pair in BAND_GATE_PAIRS],
        save_csv_path=out_dir / f"{exp_b_name}_conservative_gate.csv",
    )

    # Experiment C: recommended leakage-free held-out original-graph evaluation.
    exp_c_summary_df, exp_c_folds_df, pred_c_oof_df, exp_c_band_gate_df, exp_c_conservative_gate_df = run_heldout_original_graph_cv(
        train_df=train_df,
        test_df=test_df,
        r_cols=r_cols,
        kernel=kernel,
        cv_folds=CV_FOLDS,
        out_dir=out_dir,
    )

    exp_b_band_gate_df = exp_b_band_gate_df.assign(kernel=kernel, experiment=exp_b_name)
    exp_c_band_gate_df = exp_c_band_gate_df.assign(kernel=kernel, experiment=str(exp_c_summary_df.iloc[0]["experiment"]))
    all_band_gate_df = pd.concat([exp_b_band_gate_df, exp_c_band_gate_df], ignore_index=True)

    exp_b_conservative_gate_df = exp_b_conservative_gate_df.assign(kernel=kernel, experiment=exp_b_name)
    exp_c_conservative_gate_df = exp_c_conservative_gate_df.assign(kernel=kernel, experiment=str(exp_c_summary_df.iloc[0]["experiment"]))
    all_conservative_gate_df = pd.concat([exp_b_conservative_gate_df, exp_c_conservative_gate_df], ignore_index=True)

    master_df = pd.DataFrame([
        {
            "kernel": kernel,
            "experiment": "A_5fold_groupkfold_on_trainAug",
            "recommended_for_paper_main_table": False,
            "description": "Augmented-sample CV with base_graph groups held out inside the augmented set.",
            "train_set": f"{kernel}_train_aug",
            "test_set": "held-out augmented samples",
            "cv_oof_spearman": float(cv_summary_df.iloc[0]["oof_spearman"]),
            "cv_oof_mape": float(cv_summary_df.iloc[0]["oof_mape"]),
            "test_spearman": np.nan,
            "test_mape": np.nan,
            "test_mae_log": np.nan,
            "test_mae_speedup": np.nan,
        },
        {
            "kernel": kernel,
            "experiment": "B_trainAllAug_testOriginalSameGraphs",
            "recommended_for_paper_main_table": False,
            "description": "Old S2O/same-graph test; useful auxiliary result but has graph-identity overlap.",
            "train_set": f"{kernel}_train_aug_all_graphs",
            "test_set": f"{kernel}_test_original_same_graphs",
            "cv_oof_spearman": np.nan,
            "cv_oof_mape": np.nan,
            "test_spearman": float(exp_b_summary_df.iloc[0]["test_spearman"]),
            "test_mape": float(exp_b_summary_df.iloc[0]["test_mape"]),
            "test_mae_log": float(exp_b_summary_df.iloc[0]["test_mae_log"]),
            "test_mae_speedup": float(exp_b_summary_df.iloc[0]["test_mae_speedup"]),
        },
        {
            "kernel": kernel,
            "experiment": "C_heldoutGraphs_trainAug_testOriginal",
            "recommended_for_paper_main_table": True,
            "description": "Leakage-free unseen-graph test; train on augmented samples from training graphs, test on original held-out graphs.",
            "train_set": f"{kernel}_train_aug_training_graphs_only",
            "test_set": f"{kernel}_test_original_heldout_graphs",
            "cv_oof_spearman": np.nan,
            "cv_oof_mape": np.nan,
            "test_spearman": float(exp_c_summary_df.iloc[0]["oof_spearman"]),
            "test_mape": float(exp_c_summary_df.iloc[0]["oof_mape"]),
            "test_mae_log": float(exp_c_summary_df.iloc[0]["oof_mae_log"]),
            "test_mae_speedup": float(exp_c_summary_df.iloc[0]["oof_mae_speedup"]),
        },
    ])
    master_df.to_csv(out_dir / f"{kernel}_master_summary.csv", index=False)

    print(f"\n[{kernel}] Experiment A CV summary")
    print(cv_summary_df.to_string(index=False))
    print(f"\n[{kernel}] Experiment B same-graph test summary")
    print(exp_b_summary_df.to_string(index=False))
    print(f"\n[{kernel}] Experiment C leakage-free held-out graph summary")
    print(exp_c_summary_df.to_string(index=False))
    print(f"\n[{kernel}] All band-gate results")
    print(all_band_gate_df.to_string(index=False))
    print(f"\n[{kernel}] All conservative-gate results")
    print(all_conservative_gate_df.to_string(index=False))

    return {
        "stats": stats_df,
        "cv_summary": cv_summary_df.assign(kernel=kernel),
        "exp_b_summary": exp_b_summary_df,
        "exp_c_summary": exp_c_summary_df,
        "exp_c_folds": exp_c_folds_df,
        "band_gate": all_band_gate_df,
        "conservative_gate": all_conservative_gate_df,
        "pred": pred_b_df.assign(kernel=kernel, experiment=exp_b_name),
        "pred_heldout": pred_c_oof_df.assign(kernel=kernel, experiment=str(exp_c_summary_df.iloc[0]["experiment"])),
        "coef": coef_b_df.assign(kernel=kernel, experiment=exp_b_name),
        "master": master_df,
    }



def build_cross_kernel_outputs(results: Dict[str, Dict[str, pd.DataFrame]], out_dir: Path) -> None:
    summary_parts = []
    stats_parts = []
    band_parts = []
    gate_parts = []
    coef_parts = []
    heldout_pred_parts = []

    for kernel, dfs in results.items():
        summary_parts.append(dfs["master"])
        stats_parts.append(dfs["stats"])
        band_parts.append(dfs["band_gate"])
        gate_parts.append(dfs["conservative_gate"])
        coef_parts.append(dfs["coef"])
        heldout_pred_parts.append(dfs["pred_heldout"])

    all_master = pd.concat(summary_parts, ignore_index=True)
    all_stats = pd.concat(stats_parts, ignore_index=True)
    all_band = pd.concat(band_parts, ignore_index=True)
    all_gate = pd.concat(gate_parts, ignore_index=True)
    all_coef = pd.concat(coef_parts, ignore_index=True)
    all_heldout_pred = pd.concat(heldout_pred_parts, ignore_index=True)

    all_master.to_csv(out_dir / "all_kernels_master_summary.csv", index=False)
    all_stats.to_csv(out_dir / "all_kernels_dataset_stats.csv", index=False)
    all_band.to_csv(out_dir / "all_kernels_band_gate.csv", index=False)
    all_gate.to_csv(out_dir / "all_kernels_conservative_gate.csv", index=False)
    all_coef.to_csv(out_dir / "all_kernels_coefficients_expB_same_graph.csv", index=False)
    all_heldout_pred.to_csv(out_dir / "all_kernels_heldout_graph_oof_predictions.csv", index=False)

    # Pairwise comparison on the original test graphs when both kernels are available.
    # This keeps the old same-graph predictions for compatibility.
    if "merbit" in results and "csr" in results:
        m = results["merbit"]["pred"][[
            "graph_id", "base_graph", "speedup", "pred_speedup", "abs_err_speedup", "ape"
        ]].rename(columns={
            "speedup": "true_speedup_merbit",
            "pred_speedup": "pred_speedup_merbit",
            "abs_err_speedup": "abs_err_speedup_merbit",
            "ape": "ape_merbit",
        })
        c = results["csr"]["pred"][[
            "graph_id", "base_graph", "speedup", "pred_speedup", "abs_err_speedup", "ape"
        ]].rename(columns={
            "speedup": "true_speedup_csr",
            "pred_speedup": "pred_speedup_csr",
            "abs_err_speedup": "abs_err_speedup_csr",
            "ape": "ape_csr",
        })
        cmp_df = pd.merge(m, c, on=["graph_id", "base_graph"], how="inner")
        cmp_df["true_speedup_merbit_minus_csr"] = cmp_df["true_speedup_merbit"] - cmp_df["true_speedup_csr"]
        cmp_df["pred_speedup_merbit_minus_csr"] = cmp_df["pred_speedup_merbit"] - cmp_df["pred_speedup_csr"]
        cmp_df["merbit_true_better_than_csr"] = cmp_df["true_speedup_merbit"] > cmp_df["true_speedup_csr"]
        cmp_df.to_csv(out_dir / "test_original_same_graph_merbit_vs_csr_predictions.csv", index=False)

        cmp_summary = pd.DataFrame([{
            "num_common_graphs": int(len(cmp_df)),
            "mean_true_speedup_merbit": float(cmp_df["true_speedup_merbit"].mean()),
            "mean_true_speedup_csr": float(cmp_df["true_speedup_csr"].mean()),
            "median_true_speedup_merbit": float(cmp_df["true_speedup_merbit"].median()),
            "median_true_speedup_csr": float(cmp_df["true_speedup_csr"].median()),
            "mean_pred_speedup_merbit": float(cmp_df["pred_speedup_merbit"].mean()),
            "mean_pred_speedup_csr": float(cmp_df["pred_speedup_csr"].mean()),
            "num_merbit_true_better_than_csr": int(cmp_df["merbit_true_better_than_csr"].sum()),
            "mean_true_speedup_merbit_minus_csr": float(cmp_df["true_speedup_merbit_minus_csr"].mean()),
            "mean_pred_speedup_merbit_minus_csr": float(cmp_df["pred_speedup_merbit_minus_csr"].mean()),
        }])
        cmp_summary.to_csv(out_dir / "test_original_same_graph_merbit_vs_csr_summary.csv", index=False)

        # Pairwise comparison for leakage-free held-out predictions.
        mh = results["merbit"]["pred_heldout"][[
            "graph_id", "base_graph", "fold", "speedup", "pred_speedup", "abs_err_speedup", "ape"
        ]].rename(columns={
            "fold": "fold_merbit",
            "speedup": "true_speedup_merbit",
            "pred_speedup": "pred_speedup_merbit",
            "abs_err_speedup": "abs_err_speedup_merbit",
            "ape": "ape_merbit",
        })
        ch = results["csr"]["pred_heldout"][[
            "graph_id", "base_graph", "fold", "speedup", "pred_speedup", "abs_err_speedup", "ape"
        ]].rename(columns={
            "fold": "fold_csr",
            "speedup": "true_speedup_csr",
            "pred_speedup": "pred_speedup_csr",
            "abs_err_speedup": "abs_err_speedup_csr",
            "ape": "ape_csr",
        })
        cmp_h_df = pd.merge(mh, ch, on=["graph_id", "base_graph"], how="inner")
        cmp_h_df["true_speedup_merbit_minus_csr"] = cmp_h_df["true_speedup_merbit"] - cmp_h_df["true_speedup_csr"]
        cmp_h_df["pred_speedup_merbit_minus_csr"] = cmp_h_df["pred_speedup_merbit"] - cmp_h_df["pred_speedup_csr"]
        cmp_h_df["merbit_true_better_than_csr"] = cmp_h_df["true_speedup_merbit"] > cmp_h_df["true_speedup_csr"]
        cmp_h_df.to_csv(out_dir / "heldout_graph_merbit_vs_csr_oof_predictions.csv", index=False)

    print("\n==================== ALL KERNELS MASTER SUMMARY ====================")
    print(all_master.to_string(index=False))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RD-based predictor for MERBIT and CSR target SpMV kernels.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing input CSV files.")
    parser.add_argument("--out-dir", type=Path, default=Path("predictor_eval_merbit_csr_outputs"), help="Directory for output CSV files.")
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA, help="Ridge alpha value.")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS, help="Number of GroupKFold splits.")
    parser.add_argument("--config-json", type=Path, default=None, help="Optional JSON config mapping kernel to train_csv/test_csv.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    global RIDGE_ALPHA, CV_FOLDS
    RIDGE_ALPHA = args.ridge_alpha
    CV_FOLDS = args.cv_folds

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.config_json is not None:
        datasets = json.loads(args.config_json.read_text(encoding="utf-8"))
    else:
        datasets = DEFAULT_DATASETS

    print(f"Data dir: {data_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print(f"CV folds: {CV_FOLDS}")

    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for kernel, cfg in datasets.items():
        train_csv = data_dir / cfg["train_csv"]
        test_csv = data_dir / cfg["test_csv"]
        if not train_csv.exists():
            raise FileNotFoundError(f"Missing train CSV for {kernel}: {train_csv}")
        if not test_csv.exists():
            raise FileNotFoundError(f"Missing test CSV for {kernel}: {test_csv}")
        results[kernel] = run_one_kernel(kernel, train_csv, test_csv, out_dir)

    build_cross_kernel_outputs(results, out_dir)
    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
