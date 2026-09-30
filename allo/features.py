"""Molecular representations, kept separable so each can be ablated."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

from .config import Config

RDLogger.DisableLog("rdApp.*")

# Interpretable physicochemical block. These are exactly the properties that
# distinguish allosteric chemical space in the medicinal-chemistry literature
# (more lipophilic, fewer H-bond donors), so a model built on them alone is the
# baseline any fingerprint model has to beat.
PHYSCHEM = {
    "MolWt": Descriptors.MolWt,
    "cLogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA,
    "RotB": rdMolDescriptors.CalcNumRotatableBonds,
    "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "AliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "HeavyAtoms": lambda m: m.GetNumHeavyAtoms(),
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
    "StereoCentres": lambda m: rdMolDescriptors.CalcNumAtomStereoCenters(m),
    "FormalCharge": lambda m: Chem.GetFormalCharge(m),
    "QED": Descriptors.qed,
    "BertzCT": Descriptors.BertzCT,
    "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds,
    "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms,
    "NumBridgeheads": rdMolDescriptors.CalcNumBridgeheadAtoms,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
}


def physchem_matrix(smiles: pd.Series) -> tuple[np.ndarray, list[str]]:
    names = list(PHYSCHEM)
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append([float(PHYSCHEM[n](mol)) for n in names] if mol else [np.nan] * len(names))
    return np.asarray(rows, dtype=float), names


def ecfp_matrix(smiles: pd.Series, cfg: Config) -> tuple[np.ndarray, list[str]]:
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg.ecfp_radius, fpSize=cfg.ecfp_bits
    )
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(cfg.ecfp_bits, dtype=np.int8))
            continue
        fp = gen.GetCountFingerprintAsNumPy(mol) if cfg.ecfp_counts else gen.GetFingerprintAsNumPy(mol)
        rows.append(fp.astype(np.int8))
    names = [f"ECFP{2 * cfg.ecfp_radius}_{i}" for i in range(cfg.ecfp_bits)]
    return np.vstack(rows), names


def build_features(df: pd.DataFrame, cfg: Config, block: str = "combined"):
    """block: 'physchem' | 'ecfp' | 'combined' | 'target_only'."""
    if block == "physchem":
        return physchem_matrix(df["std_smiles"])
    if block == "ecfp":
        return ecfp_matrix(df["std_smiles"], cfg)
    if block == "target_only":
        # one-hot of the primary target: the confound baseline. If this scores
        # as well as chemistry, the model is reading target identity, not
        # binding mode.
        # astype(str) first: on a categorical column get_dummies emits one
        # column per CATEGORY, and the parquet keeps all ~8k original ChEMBL
        # target categories, so the probe would carry thousands of all-zero
        # columns.
        col = df["target_chembl_id"].astype(str)
        dummies = pd.get_dummies(col, prefix="tgt")
        return dummies.to_numpy(dtype=float), list(dummies.columns)
    if block == "combined":
        Xp, np_ = physchem_matrix(df["std_smiles"])
        Xe, ne = ecfp_matrix(df["std_smiles"], cfg)
        return np.hstack([Xp, Xe]), np_ + ne
    raise ValueError(f"unknown feature block {block!r}")
