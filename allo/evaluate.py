"""Metrics with uncertainty, and paired tests between two scorers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    pos_rate = float(np.mean(y))
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        # PR-AUC depends on prevalence; the lift over the no-skill line is the
        # comparable quantity across splits with different class balance.
        # NOTE: deliberately NOT reported as a cross-split comparable. The
        # maximum attainable lift is 1/pos_rate, which differs between splits
        # (the temporal test block has a different class balance), so a perfect
        # model would score differently in each. Kept only for within-split use.
        "pr_auc_lift": float(average_precision_score(y, p) / pos_rate),
        "mcc": float(matthews_corrcoef(y, yhat)),
        "balanced_acc": float(balanced_accuracy_score(y, yhat)),
        "n": int(len(y)),
        "pos_rate": pos_rate,
    }


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: str = "roc_auc",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Class-stratified bootstrap: resample positives and negatives separately
    so every replicate keeps the observed prevalence."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    fn = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
    }[metric]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    vals = np.asarray(vals)
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(fn(y, p)), float(lo), float(hi)


def paired_bootstrap_delta(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: str = "roc_auc",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """CI on (metric_a - metric_b) using the SAME resampled rows for both -
    the only way to ask whether model A really beats model B on this data."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}[metric]
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx]))
    deltas = np.asarray(deltas)
    lo, hi = np.quantile(deltas, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "delta": float(fn(y, p_a) - fn(y, p_b)),
        "lo": float(lo),
        "hi": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "p_two_sided": float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())),
    }


# --------------------------------------------------------------------------
# Fold-aware metrics
# --------------------------------------------------------------------------
# Pooling out-of-fold predictions into one ROC is standard practice and is
# WRONG whenever folds produce differently-calibrated scores. Under the
# target-out split each fold trains on a different set of targets, so pooling
# compares scores that were never on a common scale.
#
# The worked example is the target-identity probe: with held-out targets its
# one-hot features are all zero, so every prediction inside a fold is the same
# constant and the per-fold AUC is exactly 0.500. Pooling those constants across
# folds invents an ordering between folds and reports 0.329 - an artefact, not a
# below-chance model. Fold-mean metrics are therefore the primary numbers here.


def fold_metric(y, p, fold_ids, metric: str = "roc_auc") -> tuple[float, float, list[float]]:
    """Mean of the per-fold metric (the honest summary), plus its spread."""
    fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}[metric]
    vals = []
    for f in np.unique(fold_ids[~np.isnan(fold_ids)]):
        m = fold_ids == f
        if m.sum() > 1 and len(np.unique(y[m])) == 2:
            vals.append(fn(y[m], p[m]))
    if not vals:
        return float("nan"), float("nan"), []
    return float(np.mean(vals)), float(np.std(vals)), vals


def fold_metric_ci(
    y, p, fold_ids, metric: str = "roc_auc", n_boot: int = 2000,
    ci: float = 0.95, seed: int = 0,
) -> tuple[float, float, float]:
    """CI on the fold-mean metric, resampling compounds *within* each fold so
    the fold structure (and therefore the score scale) is preserved."""
    rng = np.random.default_rng(seed)
    fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}[metric]
    folds = [np.flatnonzero(fold_ids == f) for f in np.unique(fold_ids[~np.isnan(fold_ids)])]
    folds = [ix for ix in folds if len(np.unique(y[ix])) == 2]
    point, _, _ = fold_metric(y, p, fold_ids, metric)
    if not folds:
        return point, float("nan"), float("nan")
    draws = []
    for _ in range(n_boot):
        per = []
        for ix in folds:
            pos, neg = ix[y[ix] == 1], ix[y[ix] == 0]
            s = np.concatenate(
                [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
            )
            per.append(fn(y[s], p[s]))
        draws.append(np.mean(per))
    lo, hi = np.quantile(draws, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return point, float(lo), float(hi)


def across_fold_interval(fold_values, ci: float = 0.95) -> dict:
    """Uncertainty ACROSS folds (targets), not within them.

    `fold_metric_ci` resamples compounds inside each fold, so it captures only
    within-fold sampling noise. A claim about generalising to a *new target*
    has to carry the spread between held-out target groups, which is far wider
    and is what this returns (Student-t on the per-fold values).
    """
    from scipy import stats

    v = np.asarray([x for x in fold_values if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return {"mean": float(v[0]) if v.size else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n_folds": int(v.size)}
    mean = float(v.mean())
    sem = float(v.std(ddof=1) / np.sqrt(v.size))
    half = float(stats.t.ppf(0.5 + ci / 2, v.size - 1) * sem)
    return {"mean": mean, "lo": mean - half, "hi": mean + half,
            "sd": float(v.std(ddof=1)), "sem": sem, "n_folds": int(v.size)}


def paired_fold_delta(
    y, p_a, p_b, fold_ids, metric: str = "roc_auc", n_boot: int = 2000,
    ci: float = 0.95, seed: int = 0,
) -> dict:
    """Difference of fold-mean metrics, bootstrapped within folds on identical
    resamples for both models."""
    rng = np.random.default_rng(seed)
    fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}[metric]
    folds = [np.flatnonzero(fold_ids == f) for f in np.unique(fold_ids[~np.isnan(fold_ids)])]
    folds = [ix for ix in folds if len(np.unique(y[ix])) == 2]
    a_mean, _, a_folds = fold_metric(y, p_a, fold_ids, metric)
    b_mean, _, b_folds = fold_metric(y, p_b, fold_ids, metric)
    if not folds:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "significant": False, "p_two_sided": float("nan")}
    draws = []
    for _ in range(n_boot):
        da, db = [], []
        for ix in folds:
            pos, neg = ix[y[ix] == 1], ix[y[ix] == 0]
            s = np.concatenate(
                [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
            )
            da.append(fn(y[s], p_a[s]))
            db.append(fn(y[s], p_b[s]))
        draws.append(np.mean(da) - np.mean(db))
    draws = np.asarray(draws)
    lo, hi = np.quantile(draws, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "delta": float(a_mean - b_mean),
        "lo": float(lo),
        "hi": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "p_two_sided": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
        # 0 here means "no bootstrap draw crossed zero"; the resolution floor is
        # 1/n_boot, so report it as "< 1/n_boot" rather than as an exact zero.
        "p_resolution_floor": 1.0 / max(n_boot, 1),
        "per_fold_delta": [float(x - z) for x, z in zip(a_folds, b_folds)],
    }


def summarize(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
