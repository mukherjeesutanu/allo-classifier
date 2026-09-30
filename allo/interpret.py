"""What the model keyed on: SHAP attribution plus ECFP bit -> substructure."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from .config import Config
from .features import build_features
from .models import make_model
from .splits import get_splits

RDLogger.DisableLog("rdApp.*")


def shap_importance(df: pd.DataFrame, cfg: Config, block: str = "combined",
                    scheme: str = "target", top_n: int = 25) -> pd.DataFrame:
    """Mean |SHAP| over out-of-fold test rows, so importances are measured on
    molecules the model never saw - not on its own training set."""
    import shap

    X, names = build_features(df, cfg, block)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy()
    splits = get_splits(df, scheme, cfg.n_folds, cfg.random_state)

    total = np.zeros(X.shape[1])
    signed = np.zeros(X.shape[1])
    # A fingerprint bit is 0 in most molecules, and SHAP still attributes to
    # that absence. Averaging the sign over ALL rows therefore describes the
    # bit being *missing*, which inverts the reported direction for rare bits.
    # Sign is accumulated only where the feature is actually present.
    signed_present = np.zeros(X.shape[1])
    n_present = np.zeros(X.shape[1])
    n_scored = 0
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        model = make_model("lightgbm", cfg)
        model.fit(X[tr], y[tr])
        vals = shap.TreeExplainer(model).shap_values(X[te])
        if isinstance(vals, list):          # binary LightGBM -> [neg, pos]
            vals = vals[1]
        elif vals.ndim == 3:                # newer shap -> (rows, features, classes)
            vals = vals[..., 1]
        total += np.abs(vals).sum(axis=0)
        signed += vals.sum(axis=0)
        present = X[te] > 0
        signed_present += np.where(present, vals, 0.0).sum(axis=0)
        n_present += present.sum(axis=0)
        n_scored += len(te)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_present = np.where(n_present > 0, signed_present / n_present, np.nan)
    imp = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": total / max(n_scored, 1),
            "mean_signed_shap": signed / max(n_scored, 1),
            "mean_signed_shap_when_present": mean_present,
            "n_molecules_with_feature": n_present.astype(int),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    # Direction describes what the feature being PRESENT does to the prediction.
    imp["direction"] = np.where(
        imp["mean_signed_shap_when_present"] > 0, "-> allosteric", "-> orthosteric"
    )
    return imp.head(top_n).reset_index(drop=True)


def bit_enrichment(df: pd.DataFrame, cfg: Config, bits: list[int]) -> pd.DataFrame:
    """Per-bit prevalence in each class - the plain-numbers check on SHAP."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=cfg.ecfp_radius, fpSize=cfg.ecfp_bits)
    mols = [Chem.MolFromSmiles(s) for s in df["std_smiles"]]
    fps = np.vstack([
        gen.GetFingerprintAsNumPy(m) if m is not None else np.zeros(cfg.ecfp_bits, dtype=np.int8)
        for m in mols
    ])
    y = df["label"].to_numpy()
    rows = []
    for b in bits:
        col = fps[:, b].astype(bool)
        p_allo = col[y == 1].mean() if (y == 1).any() else np.nan
        p_orth = col[y == 0].mean() if (y == 0).any() else np.nan
        rows.append(
            {
                "bit": b,
                "frac_in_allosteric": float(p_allo),
                "frac_in_orthosteric": float(p_orth),
                "enrichment": float((p_allo + 1e-9) / (p_orth + 1e-9)),
                "n_molecules": int(col.sum()),
                "example_smiles": next(
                    (df["std_smiles"].iloc[i] for i in np.flatnonzero(col)[:1]), ""
                ),
            }
        )
    return pd.DataFrame(rows)


def bit_substructures(df: pd.DataFrame, cfg: Config, bit: int, max_examples: int = 3) -> list[str]:
    """SMARTS-ish sketch of what an ECFP bit actually matches, by pulling the
    atom environment out of the first few molecules that set it."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=cfg.ecfp_radius, fpSize=cfg.ecfp_bits)
    out = []
    for smi in df["std_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=ao)
        info = ao.GetBitInfoMap()
        if bit not in info:
            continue
        atom, radius = info[bit][0]
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, max(radius, 1), atom)
        amap: dict[int, int] = {}
        sub = Chem.PathToSubmol(mol, env, atomMap=amap)
        frag = Chem.MolToSmiles(sub, canonical=True) if sub.GetNumAtoms() else ""
        if frag:
            out.append(frag)
        if len(out) >= max_examples:
            break
    return out
