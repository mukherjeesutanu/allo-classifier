"""Cross-validation schemes, ordered by how hard they are to cheat.

random      - optimistic; reported only to show the gap the others expose
scaffold    - no generic Bemis-Murcko scaffold spans train and test
target      - no target spans train and test (the mechanism-generalisation test)
temporal    - train on older literature, test on newer: the prospective setting

The scaffold and target schemes are the honest ones. `target` is the decisive
experiment for this project: allosteric annotation is concentrated in a few
receptor families, so a model that merely memorises "mGlu chemotype => allosteric"
collapses here while a model that has learned transferable chemistry does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Single chronological holdout on first-publication year.

    Compounds are ordered by the year they first appear in the literature; the
    most recent `test_frac` become the test set. The cut is placed on a year
    boundary, so no publication year straddles train and test.
    """
    if "first_year" not in df:
        raise KeyError("temporal split needs a 'first_year' column")
    # Compounds with no publication year cannot be placed chronologically.
    # They are imputed to the median and therefore land in TRAIN, so the
    # temporal test block covers only compounds that carry a year.
    missing = int(df["first_year"].isna().sum())
    if missing:
        import warnings
        warnings.warn(
            f"temporal split: {missing} of {len(df)} compounds "
            f"({missing / len(df):.1%}) have no publication year and are "
            f"imputed into the training side",
            stacklevel=2,
        )
    years = df["first_year"].fillna(df["first_year"].median())
    cut = np.quantile(years, 1 - test_frac)
    train = np.flatnonzero((years <= cut).to_numpy())
    test = np.flatnonzero((years > cut).to_numpy())
    return [(train, test)]


def get_splits(df: pd.DataFrame, scheme: str, n_folds: int, random_state: int):
    y = df["label"].to_numpy()
    if scheme == "random":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return list(cv.split(np.zeros(len(df)), y))
    if scheme == "temporal":
        return temporal_split(df)
    if scheme == "scaffold":
        groups = df["scaffold"].to_numpy()
    elif scheme == "target":
        groups = df["target_chembl_id"].to_numpy()
    else:
        raise ValueError(f"unknown split scheme {scheme!r}")

    # StratifiedGroupKFold keeps the class ratio balanced across folds while
    # honouring the group constraint; it falls back to GroupKFold when a scheme
    # makes stratification impossible (e.g. single-class groups dominate).
    try:
        cv = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
        splits = list(cv.split(np.zeros(len(df)), y, groups))
        if all(len(np.unique(y[te])) == 2 for _, te in splits):
            return splits
    except ValueError:
        pass
    cv = GroupKFold(n_splits=n_folds)
    return list(cv.split(np.zeros(len(df)), y, groups))


def split_report(df: pd.DataFrame, splits, scheme: str) -> pd.DataFrame:
    rows = []
    for i, (tr, te) in enumerate(splits):
        rows.append(
            {
                "scheme": scheme,
                "fold": i,
                "n_train": len(tr),
                "n_test": len(te),
                "test_pos": int(df["label"].to_numpy()[te].sum()),
                "test_pos_frac": float(df["label"].to_numpy()[te].mean()),
                "n_train_targets": int(pd.unique(df["target_chembl_id"].to_numpy()[tr]).size),
                "n_test_targets": int(pd.unique(df["target_chembl_id"].to_numpy()[te]).size),
            }
        )
    return pd.DataFrame(rows)


def leakage_check(df: pd.DataFrame, splits, scheme: str) -> dict:
    """Assert the group constraint actually holds (guards against silent bugs)."""
    if scheme == "temporal":
        ok = all(
            df["first_year"].iloc[tr].max() <= df["first_year"].iloc[te].min()
            for tr, te in splits
        )
        return {"scheme": scheme, "checked": True, "leak_free": bool(ok)}
    col = {"scaffold": "scaffold", "target": "target_chembl_id"}.get(scheme)
    if col is None:
        return {"scheme": scheme, "checked": False}
    overlaps = []
    vals = df[col].to_numpy()
    for tr, te in splits:
        overlaps.append(len(set(vals[tr]) & set(vals[te])))
    return {"scheme": scheme, "checked": True, "group_overlaps": overlaps,
            "leak_free": all(o == 0 for o in overlaps)}
