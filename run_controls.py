#!/usr/bin/env python
"""Negative controls on the real dataset.

Two questions the main benchmark cannot answer about itself:

1. **Label permutation.** Shuffle the labels and rerun the hardest split. Any
   score above chance is machinery artefact, not chemistry.
2. **Pooled-vs-fold metric.** Quantify the inflation from pooling out-of-fold
   predictions across folds, using the target-identity probe whose true
   per-fold AUC is known analytically to be 0.500.

    python run_controls.py [--negatives all|strict] [--n-permutations 5]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from allo.config import CFG, OUTPUTS, PROCESSED
from allo.evaluate import fold_metric
from allo.pipeline import feature_cache, run_cell
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", choices=["all", "strict"], default="strict")
    ap.add_argument("--n-permutations", type=int, default=5)
    ap.add_argument("--split", default="target")
    args = ap.parse_args()

    suffix = "" if args.negatives == "all" else f"_{args.negatives}"
    df = pd.read_parquet(PROCESSED / f"dataset{suffix}.parquet")
    cache = feature_cache(df, CFG, blocks=("combined", "target_only"))
    rng = np.random.default_rng(CFG.random_state)

    report: dict = {"dataset": f"dataset{suffix}", "split": args.split, "n": len(df)}

    # --- 1. true performance, for reference -----------------------------
    oof, y, _, folds = run_cell(df, CFG, "combined", "lightgbm", args.split, cache)
    true_mean, true_sd, true_folds = fold_metric(y, oof, folds)
    report["true_fold_auc_mean"] = true_mean
    report["true_fold_auc_sd"] = true_sd
    report["true_per_fold"] = [round(v, 4) for v in true_folds]
    print(f"real labels     fold-AUC = {true_mean:.3f} +- {true_sd:.3f}")

    # --- 2. label permutation -------------------------------------------
    perm_means = []
    shuffled = df.copy()
    for i in range(args.n_permutations):
        shuffled["label"] = rng.permutation(df["label"].to_numpy())
        oof_p, y_p, _, folds_p = run_cell(
            shuffled, CFG, "combined", "lightgbm", args.split, cache
        )
        m, _, _ = fold_metric(y_p, oof_p, folds_p)
        perm_means.append(m)
        print(f"permutation {i + 1}   fold-AUC = {m:.3f}")
    report["permuted_fold_auc"] = [round(v, 4) for v in perm_means]
    report["permuted_mean"] = float(np.mean(perm_means))
    report["permuted_max"] = float(np.max(perm_means))
    # one-sided empirical p: how often noise reaches the real score
    report["p_permutation"] = float(
        (np.sum(np.asarray(perm_means) >= true_mean) + 1) / (len(perm_means) + 1)
    )

    # --- 3. pooling artefact --------------------------------------------
    oof_t, y_t, _, folds_t = run_cell(df, CFG, "target_only", "logistic", args.split, cache)
    mask = ~np.isnan(oof_t)
    pooled = float(roc_auc_score(y_t[mask], oof_t[mask]))
    fold_mean, _, _ = fold_metric(y_t, oof_t, folds_t)
    report["target_probe_pooled_auc"] = pooled
    report["target_probe_fold_auc"] = fold_mean
    report["pooling_bias"] = pooled - fold_mean
    print(
        f"target-identity probe: pooled AUC {pooled:.3f} vs fold-mean "
        f"{fold_mean:.3f} (analytic truth 0.500)"
    )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"controls{suffix}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote outputs/controls{suffix}.json")


if __name__ == "__main__":
    main()
