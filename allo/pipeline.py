"""Benchmark orchestration: every (feature block x model x split scheme) cell."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import (
    bootstrap_ci,
    across_fold_interval,
    fold_metric,
    fold_metric_ci,
    paired_fold_delta,
    point_metrics,
)
from .features import build_features
from .models import fit_predict, make_model
from .splits import get_splits, leakage_check

SCHEMES = ("random", "scaffold", "temporal", "target")
BLOCKS = ("target_only", "physchem", "ecfp", "combined")


def feature_cache(df: pd.DataFrame, cfg: Config, blocks=BLOCKS) -> dict:
    """Featurise each block exactly once.

    Without this every (block, model, split) cell re-runs RDKit over the whole
    set; the physicochemical block alone (BertzCT, QED) then dominates runtime.
    """
    cache = {}
    for block in blocks:
        X, names = build_features(df, cfg, block)
        cache[block] = (np.ascontiguousarray(X), names)
    return cache


def run_cell(
    df: pd.DataFrame,
    cfg: Config,
    block: str,
    model_name: str,
    scheme: str,
    cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Out-of-fold predictions for one benchmark cell.

    Also returns, for every row, the index of the fold that predicted it -
    metrics must be computed inside folds, not on the pooled vector.
    """
    X, names = (cache or {}).get(block) or build_features(df, cfg, block)
    y = df["label"].to_numpy()
    splits = get_splits(df, scheme, cfg.n_folds, cfg.random_state)

    oof = np.full(len(df), np.nan)
    fold_ids = np.full(len(df), np.nan)
    per_fold = []
    for i, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2:
            continue
        model = make_model(model_name, cfg)
        p = fit_predict(model, X[tr], y[tr], X[te], names)
        oof[te] = p
        fold_ids[te] = i
        if len(np.unique(y[te])) == 2:
            per_fold.append({"fold": i, **point_metrics(y[te], p)})
    return oof, y, per_fold, fold_ids


def run_benchmark(df: pd.DataFrame, cfg: Config, models=("lightgbm", "logistic"),
                  verbose: bool = True) -> dict:
    results, oof_store, fold_store = [], {}, {}
    cache = feature_cache(df, cfg)
    if verbose:
        print(f"  featurised {len(df):,} molecules", flush=True)

    for scheme in SCHEMES:
        splits = get_splits(df, scheme, cfg.n_folds, cfg.random_state)
        leak = leakage_check(df, splits, scheme)
        if leak.get("checked") and not leak["leak_free"]:
            raise RuntimeError(f"group leakage in {scheme} split: {leak}")

        for block in BLOCKS:
            for model_name in models:
                # target_only is a confound probe, not a chemistry model; one
                # estimator is enough and logistic is the honest choice there.
                if block == "target_only" and model_name != "logistic":
                    continue
                t0 = time.time()
                oof, y, per_fold, fold_ids = run_cell(
                    df, cfg, block, model_name, scheme, cache
                )
                mask = ~np.isnan(oof)
                if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                    continue
                # primary: mean of per-fold AUCs, CI bootstrapped within folds
                f_auc, f_lo, f_hi = fold_metric_ci(
                    y, oof, fold_ids, "roc_auc", cfg.n_bootstrap,
                    cfg.bootstrap_ci, seed=cfg.random_state,
                )
                f_pr, fpr_lo, fpr_hi = fold_metric_ci(
                    y, oof, fold_ids, "pr_auc", cfg.n_bootstrap,
                    cfg.bootstrap_ci, seed=cfg.random_state,
                )
                _, auc_sd, fold_aucs = fold_metric(y, oof, fold_ids, "roc_auc")
                across = across_fold_interval(fold_aucs, cfg.bootstrap_ci)
                # secondary: the pooled figure, kept only for comparison
                auc, lo, hi = bootstrap_ci(
                    y[mask], oof[mask], "roc_auc", cfg.n_bootstrap, cfg.bootstrap_ci,
                    seed=cfg.random_state,
                )
                m = point_metrics(y[mask], oof[mask])
                if verbose:
                    print(f"  {scheme:9s} {block:12s} {model_name:9s} "
                          f"foldAUC={f_auc:.3f}+-{auc_sd:.3f} "
                          f"pooled={m['roc_auc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
                results.append(
                    {
                        "split": scheme,
                        "features": block,
                        "model": model_name,
                        **m,
                        "fold_roc_auc": f_auc,
                        "fold_roc_auc_lo": f_lo,
                        "fold_roc_auc_hi": f_hi,
                        "fold_roc_auc_sd": auc_sd,
                        "fold_pr_auc": f_pr,
                        "fold_pr_auc_lo": fpr_lo,
                        "fold_pr_auc_hi": fpr_hi,
                        "pooled_roc_auc": m["roc_auc"],
                        "pooled_roc_auc_lo": lo,
                        "pooled_roc_auc_hi": hi,
                        "across_fold_lo": across.get("lo", np.nan),
                        "across_fold_hi": across.get("hi", np.nan),
                        "per_fold_auc": [round(v, 4) for v in fold_aucs],
                        "n_folds_scored": len(per_fold),
                    }
                )
                oof_store[(scheme, block, model_name)] = oof
                fold_store[(scheme, block, model_name)] = fold_ids

    res = pd.DataFrame(results)

    # --- head-to-head comparisons that decide the scientific claim --------
    comparisons = []
    y = df["label"].to_numpy()
    for scheme in SCHEMES:
        pairs = [
            ("combined", "lightgbm", "physchem", "lightgbm", "structure beyond physchem"),
            ("combined", "lightgbm", "target_only", "logistic", "chemistry beyond target identity"),
            ("ecfp", "lightgbm", "physchem", "lightgbm", "fingerprint beyond physchem"),
        ]
        for ba, ma, bb, mb, question in pairs:
            ka, kb = (scheme, ba, ma), (scheme, bb, mb)
            if ka not in oof_store or kb not in oof_store:
                continue
            # Under target-out the probe is constant within every fold, so its
            # AUC is exactly 0.5 by construction and the "delta" would just
            # restate that the model beats chance - which the permutation
            # control already establishes. Not a head-to-head; skip it.
            if scheme == "target" and bb == "target_only":
                continue
            # Both cells share the same split, so they share fold ids; the
            # delta is between fold-mean AUCs, resampled inside folds.
            d = paired_fold_delta(
                y, oof_store[ka], oof_store[kb], fold_store[ka],
                "roc_auc", cfg.n_bootstrap, cfg.bootstrap_ci, seed=cfg.random_state,
            )
            comparisons.append(
                {"split": scheme, "question": question, "a": f"{ba}/{ma}",
                 "b": f"{bb}/{mb}", **d}
            )

    return {
        "results": res,
        "comparisons": pd.DataFrame(comparisons),
        "oof": oof_store,
        "folds": fold_store,
    }
