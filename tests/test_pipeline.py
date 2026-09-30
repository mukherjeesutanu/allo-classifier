"""Guards on the machinery itself.

The decisive one is `test_shuffled_labels_give_chance_performance`: if the
benchmark can score above chance on labels that carry no information, every
number the project reports is worthless. It runs on synthetic molecules so it
needs no ChEMBL download.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, RDLogger

from allo.config import Config
from allo.dataset import build_dataset, generic_scaffold, standardize
from allo.evaluate import fold_metric
from allo.pipeline import feature_cache, run_cell
from allo.splits import get_splits, leakage_check

RDLogger.DisableLog("rdApp.*")

FRAGMENTS = [
    "c1ccccc1", "C1CCNCC1", "c1ccncc1", "C1CCOCC1", "c1cc[nH]c1",
    "C1CCCCC1", "c1ccsc1", "CC(=O)N", "CS(=O)(=O)N", "C(F)(F)F", "OCC",
]


def _molecules(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out: set[str] = set()
    while len(out) < n:
        smi = "".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(2, 4)))
        mol = Chem.MolFromSmiles(smi)
        if mol and 8 <= mol.GetNumHeavyAtoms() <= 70:
            out.add(Chem.MolToSmiles(mol))
    return sorted(out)


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    rng = random.Random(1)
    smis = _molecules(400)
    targets = [f"CHEMBL{100 + i}" for i in range(8)]
    rows = []
    for i, smi in enumerate(smis):
        target = rng.choice(targets)
        allosteric = rng.random() < 0.4
        desc = (
            f"Positive allosteric modulation of {target} by FLIPR assay"
            if allosteric
            else f"Competitive inhibition of {target}"
        )
        rows.append(
            dict(
                compound_chembl_id=f"C{i}", smiles=smi, target_chembl_id=target,
                target_name=target, target_family="fam", assay_chembl_id=f"A{i % 50}",
                assay_type="B", assay_description=desc, standard_type="IC50",
                standard_relation="=", standard_value=100.0, standard_units="nM",
                pchembl_value=7.0, data_validity_comment=None,
                potential_duplicate=0, year=2020,
            )
        )
    df, _ = build_dataset(pd.DataFrame(rows), Config())
    return df


def test_dataset_is_wellformed(synthetic):
    df = synthetic
    assert len(df) > 50
    assert set(df["label"]) <= {0, 1}
    assert df["inchikey"].is_unique
    assert df["std_smiles"].map(lambda s: Chem.MolFromSmiles(s) is not None).all()


def test_matched_target_design_holds_in_final_table(synthetic):
    """Every retained target must carry >=k compounds of BOTH classes in the
    table the benchmark actually uses. Enforcing this on compound-target
    incidences before collapsing to one target per compound does not survive
    the collapse, which is the bug this pins."""
    k = Config().min_compounds_per_class_per_target
    counts = synthetic.groupby("target_chembl_id", observed=True)["label"].agg(
        pos=lambda s: int((s == 1).sum()), neg=lambda s: int((s == 0).sum())
    )
    assert len(counts), "no targets survived"
    assert (counts["pos"] >= k).all(), f"targets short of positives:\n{counts}"
    assert (counts["neg"] >= k).all(), f"targets short of negatives:\n{counts}"


def test_scaffold_split_has_no_shared_scaffold(synthetic):
    splits = get_splits(synthetic, "scaffold", 5, 42)
    assert leakage_check(synthetic, splits, "scaffold")["leak_free"]


def test_target_split_has_no_shared_target(synthetic):
    splits = get_splits(synthetic, "target", 5, 42)
    assert leakage_check(synthetic, splits, "target")["leak_free"]


def test_shuffled_labels_give_chance_performance(synthetic):
    """Null control: permuted labels must not be learnable."""
    cfg = Config()
    df = synthetic.copy()
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(3):
        df["label"] = rng.permutation(df["label"].to_numpy())
        oof, y, _, folds = run_cell(df, cfg, "combined", "lightgbm", "scaffold")
        mean, _, vals = fold_metric(y, oof, folds)
        if vals:
            aucs.append(mean)
    assert aucs, "no scorable fold"
    assert max(aucs) < 0.65, f"pipeline finds signal in noise: {aucs}"


def test_constant_predictions_score_exactly_chance_per_fold(synthetic):
    """Guards the metric itself: a model with no information must score 0.5
    per fold. Pooling such predictions across folds does NOT, which is why
    fold-mean is the reported metric."""
    cfg = Config()
    y = synthetic["label"].to_numpy()
    oof, y2, _, folds = run_cell(synthetic, cfg, "target_only", "logistic", "target")
    mean, _, vals = fold_metric(y2, oof, folds)
    assert vals, "no scorable fold"
    assert abs(mean - 0.5) < 1e-9, f"uninformative probe scored {mean}"


def test_feature_cache_matches_uncached(synthetic):
    cfg = Config()
    cache = feature_cache(synthetic, cfg, blocks=("physchem",))
    a, _, _, _ = run_cell(synthetic, cfg, "physchem", "lightgbm", "scaffold")
    b, _, _, _ = run_cell(synthetic, cfg, "physchem", "lightgbm", "scaffold", cache)
    np.testing.assert_allclose(a, b)


def test_standardize_desalts():
    smi, key = standardize("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
    assert smi is not None and "Na" not in smi and key


def test_generic_scaffold_groups_analogues():
    """Same ring system, different decoration -> same generic scaffold."""
    a = generic_scaffold("c1ccc(-c2ccncc2)cc1")
    b = generic_scaffold("Clc1ccc(-c2ccncc2)cc1")
    assert a == b and a
