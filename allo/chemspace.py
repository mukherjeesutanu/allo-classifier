"""Marginal vs within-target physicochemical comparison.

The medicinal-chemistry folklore is that allosteric modulators are more
lipophilic, less polar and more three-dimensional than orthosteric ligands.
Testing it by pooling all compounds is exactly the mistake this project is about:
the two classes are drawn from different target families, so a pooled difference
confounds binding mode with target.

`property_deltas` computes both readings. Where they disagree in sign the
comparison is a Simpson's paradox and only the within-target number is about
binding mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .config import Config
from .features import PHYSCHEM

RDLogger.DisableLog("rdApp.*")

POS, NEG = 1, 0


def property_table(df: pd.DataFrame) -> pd.DataFrame:
    mols = [Chem.MolFromSmiles(s) for s in df["std_smiles"]]
    data = {
        name: [float(fn(m)) if m is not None else np.nan for m in mols]
        for name, fn in PHYSCHEM.items()
    }
    out = pd.DataFrame(data, index=df.index)
    out["label"] = df["label"].to_numpy()
    out["target_chembl_id"] = df["target_chembl_id"].to_numpy()
    return out


def property_deltas(df: pd.DataFrame, cfg: Config, min_per_class: int | None = None) -> pd.DataFrame:
    """Median(allosteric) - median(orthosteric), pooled and within-target.

    The within-target figure averages the per-target difference over targets
    that carry at least `min_per_class` compounds of each class, so every
    comparison is made against ligands of the *same* protein.
    """
    k = min_per_class or cfg.min_compounds_per_class_per_target
    tab = property_table(df)
    props = list(PHYSCHEM)

    rows = []
    counts = tab.groupby("target_chembl_id", observed=True)["label"].agg(
        n_pos=lambda s: int((s == POS).sum()), n_neg=lambda s: int((s == NEG).sum())
    )
    usable = counts[(counts["n_pos"] >= k) & (counts["n_neg"] >= k)].index

    for p in props:
        a = tab.loc[tab["label"] == POS, p]
        o = tab.loc[tab["label"] == NEG, p]
        pooled = float(np.nanmedian(a) - np.nanmedian(o))
        # Properties live on wildly different scales (BertzCT ~1e2, cLogP ~1e0),
        # so raw deltas cannot be compared or plotted together. Dividing by the
        # spread of the property makes every shift a standardised effect size.
        spread = float(np.nanstd(tab[p].to_numpy(dtype=float)))
        spread = spread if spread > 1e-12 else np.nan

        per_target = []
        for t in usable:
            sub = tab[tab["target_chembl_id"] == t]
            da = np.nanmedian(sub.loc[sub["label"] == POS, p])
            do = np.nanmedian(sub.loc[sub["label"] == NEG, p])
            if np.isfinite(da) and np.isfinite(do):
                per_target.append(da - do)
        per_target = np.asarray(per_target, dtype=float)
        if per_target.size:
            mean_within = float(per_target.mean())
            sd = float(per_target.std(ddof=1)) if per_target.size > 1 else np.nan
            sem = sd / np.sqrt(per_target.size) if np.isfinite(sd) else np.nan
            lo, hi = (mean_within - 1.96 * sem, mean_within + 1.96 * sem) if np.isfinite(sem) else (np.nan, np.nan)
            frac_pos = float((per_target > 0).mean())
        else:
            mean_within = sd = lo = hi = frac_pos = np.nan

        rows.append(
            {
                "property": p,
                "pooled_delta": pooled,
                "property_sd": spread,
                "pooled_delta_std": pooled / spread,
                "within_target_delta_std": mean_within / spread,
                "within_lo_std": lo / spread,
                "within_hi_std": hi / spread,
                "within_target_delta": mean_within,
                "within_target_median": float(np.median(per_target)) if per_target.size else np.nan,
                "within_lo": lo,
                "within_hi": hi,
                "n_targets": int(per_target.size),
                "frac_targets_positive": frac_pos,
                "sign_flip": bool(
                    per_target.size
                    and np.isfinite(pooled)
                    and np.isfinite(mean_within)
                    and np.sign(pooled) != np.sign(mean_within)
                    and abs(pooled) > 1e-9
                    and abs(mean_within) > 1e-9
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "within_target_delta_std", key=lambda s: -s.abs()
    ).reset_index(drop=True)
