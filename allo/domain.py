"""Applicability domain: how far test molecules sit from the training set.

A split scheme is only 'harder' if it actually pushes test compounds away from
what the model saw. Measuring nearest-neighbour Tanimoto similarity turns the
performance gap between split schemes from an assertion into a measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from .config import Config
from .splits import get_splits

RDLogger.DisableLog("rdApp.*")


def _fingerprints(smiles: pd.Series, cfg: Config):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg.ecfp_radius, fpSize=cfg.ecfp_bits
    )
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    return fps


def nn_similarity_by_split(df: pd.DataFrame, cfg: Config, schemes=("random", "scaffold", "temporal", "target")) -> pd.DataFrame:
    """Max Tanimoto from each test compound to any training compound."""
    fps = _fingerprints(df["std_smiles"], cfg)
    rows = []
    for scheme in schemes:
        try:
            splits = get_splits(df, scheme, cfg.n_folds, cfg.random_state)
        except KeyError:
            continue
        sims: list[float] = []
        for tr, te in splits:
            train_fps = [fps[i] for i in tr if fps[i] is not None]
            if not train_fps:
                continue
            for i in te:
                if fps[i] is None:
                    continue
                sims.append(max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)))
        if not sims:
            continue
        s = np.asarray(sims)
        rows.append(
            {
                "split": scheme,
                "median_nn_tanimoto": float(np.median(s)),
                "mean_nn_tanimoto": float(s.mean()),
                "frac_above_0.7": float((s > 0.7).mean()),
                "frac_above_0.4": float((s > 0.4).mean()),
                "n_test_compounds": int(len(s)),
            }
        )
    return pd.DataFrame(rows)
