#!/usr/bin/env python
"""Build the benchmark dataset from the ChEMBL SQLite release.

    python run_build_dataset.py [--negatives all|strict] [--sqlite PATH]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from allo.chembl import load_activities, target_summary
from allo.config import CFG, PROCESSED, OUTPUTS
from allo.dataset import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", choices=["all", "strict"], default="all")
    ap.add_argument("--sqlite", default=None)
    ap.add_argument("--cache", default=str(PROCESSED / "activities_raw.parquet"))
    ap.add_argument(
        "--neg-ratio",
        type=float,
        default=None,
        help="cap negatives at RATIO x positives within each target "
             "(weak-negative variant only; keeps the full set tractable)",
    )
    ap.add_argument("--tag", default=None, help="suffix for the output files")
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache)
    if cache.exists():
        print(f"[1/3] reading cached activity table {cache}")
        acts = pd.read_parquet(cache)
    else:
        print("[1/3] querying ChEMBL SQLite (a few minutes)…")
        t0 = time.time()
        acts = load_activities(CFG, args.sqlite)
        print(f"      {len(acts):,} rows in {time.time() - t0:.0f}s")
        acts.to_parquet(cache, index=False)

    print("[2/3] building labelled compound set…")
    df, prov = build_dataset(
        acts, CFG, negatives=args.negatives, neg_ratio=args.neg_ratio
    )

    suffix = args.tag or ("" if args.negatives == "all" else f"_{args.negatives}")
    if args.tag and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    out = PROCESSED / f"dataset{suffix}.parquet"
    df.to_parquet(out, index=False)
    prov.to_json(OUTPUTS / f"dataset_provenance{suffix}.json")
    # per-target composition of the FINAL benchmark table (the raw-ChEMBL
    # summary describes a different population and would be misleading here)
    (
        df.groupby(["target_chembl_id", "target_name"], observed=True)["label"]
        .agg(n_compounds="size",
             n_allosteric=lambda s: int((s == 1).sum()),
             n_orthosteric=lambda s: int((s == 0).sum()))
        .assign(frac_allosteric=lambda t: t["n_allosteric"] / t["n_compounds"])
        .sort_values("n_compounds", ascending=False)
        .to_csv(OUTPUTS / f"target_composition{suffix}.csv")
    )

    print(f"[3/3] wrote {out}")
    print(f"      {len(df):,} compounds | "
          f"{int(df['label'].sum()):,} allosteric / {int((df['label'] == 0).sum()):,} orthosteric")
    print(f"      {df['target_chembl_id'].nunique()} targets, "
          f"{df['scaffold'].nunique():,} generic scaffolds")
    print(df["tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
