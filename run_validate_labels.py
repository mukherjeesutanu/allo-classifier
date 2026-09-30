#!/usr/bin/env python
"""Measure how well the assay-text rules recover mechanism annotation.

Everything here runs against the local ChEMBL pull, so the numbers in the
README are reproducible rather than quoted from an ad-hoc query.

Three measurements:

1. **Recall** on assays whose description contains the literal word
   "allosteric" — how many does `classify_description` actually call
   ALLOSTERIC, and what are the misses?
2. **Specificity** on a random sample of assays that do *not* contain it —
   how often does the rule fire anyway?
3. **Manual audit sample** — a reproducible random sample of positives written
   to CSV so the calls can be checked by eye.

    python run_validate_labels.py [--sample 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from allo.config import CFG, OUTPUTS, PROCESSED
from allo.label import ALLOSTERIC, classify_description

WORD = re.compile(r"\ballosteric", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=str(PROCESSED / "activities_raw.parquet"))
    args = ap.parse_args()

    acts = pd.read_parquet(args.cache, columns=["assay_chembl_id", "assay_description"])
    assays = acts.drop_duplicates("assay_chembl_id").reset_index(drop=True)
    assays["evidence"] = assays["assay_description"].map(classify_description)
    assays["mentions"] = assays["assay_description"].fillna("").map(lambda s: bool(WORD.search(s)))

    rep: dict = {"n_unique_assays": int(len(assays))}

    # --- 1. recall on assays that mention the word ----------------------
    mention = assays[assays["mentions"]]
    called = int((mention["evidence"] == ALLOSTERIC).sum())
    rep["n_assays_mentioning_allosteric"] = int(len(mention))
    rep["recall_on_mentions"] = called / max(len(mention), 1)
    misses = mention[mention["evidence"] != ALLOSTERIC]
    rep["n_misses"] = int(len(misses))
    rep["miss_breakdown"] = misses["evidence"].value_counts().to_dict()
    misses[["assay_chembl_id", "evidence", "assay_description"]].to_csv(
        OUTPUTS / "label_validation_misses.csv", index=False
    )

    # --- 2. specificity on assays that never mention it -----------------
    rest = assays[~assays["mentions"]]
    fired = rest[rest["evidence"] == ALLOSTERIC]
    rep["n_assays_without_the_word"] = int(len(rest))
    rep["n_fired_without_the_word"] = int(len(fired))
    rep["false_positive_rate_without_the_word"] = len(fired) / max(len(rest), 1)

    rng = np.random.default_rng(args.seed)
    n = min(args.sample, len(rest))
    idx = rng.choice(len(rest), n, replace=False)
    samp = rest.iloc[idx]
    rep["random_sample_size"] = int(n)
    rep["random_sample_called_allosteric"] = int((samp["evidence"] == ALLOSTERIC).sum())

    # --- 3. audit sample of positive calls ------------------------------
    pos = assays[assays["evidence"] == ALLOSTERIC]
    take = min(200, len(pos))
    audit = pos.iloc[rng.choice(len(pos), take, replace=False)]
    audit[["assay_chembl_id", "assay_description"]].to_csv(
        OUTPUTS / "label_validation_audit_sample.csv", index=False
    )
    rep["n_assays_called_allosteric"] = int(len(pos))
    rep["audit_sample_written"] = take

    rep["evidence_distribution"] = assays["evidence"].value_counts().to_dict()

    (OUTPUTS / "label_validation.json").write_text(json.dumps(rep, indent=2, default=int))
    print(json.dumps(rep, indent=2, default=int))
    print("\nwrote outputs/label_validation.json, label_validation_misses.csv, "
          "label_validation_audit_sample.csv")


if __name__ == "__main__":
    main()
