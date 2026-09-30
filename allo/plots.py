"""Figures for the report. One idea per panel, readable in greyscale."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
    }
)

TEAL, CORAL, SLATE, AMBER = "#2a9d8f", "#e76f51", "#4a5568", "#e9c46a"


def fig_split_gap(res: pd.DataFrame, path: Path) -> None:
    """The headline figure: how much performance is split-scheme artefact."""
    sub = res[(res["model"] == "lightgbm") & (res["features"] == "combined")]
    order = ["random", "scaffold", "temporal", "target"]
    labels = ["random\nsplit", "scaffold-out", "temporal\n(prospective)", "target-out"]
    sub = sub.set_index("split").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    x = np.arange(len(sub))
    err = np.vstack([sub["fold_roc_auc"] - sub["fold_roc_auc_lo"],
                     sub["fold_roc_auc_hi"] - sub["fold_roc_auc"]])
    ax.bar(x, sub["fold_roc_auc"], yerr=err, capsize=4,
           color=[TEAL, AMBER, "#8ab17d", CORAL], width=0.6)
    ax.axhline(0.5, ls="--", lw=1, color=SLATE, label="random classifier")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean per-fold ROC-AUC")
    ax.set_ylim(0.4, 1.08)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title("Generalisation falls as the split gets honest", loc="left", fontsize=10)
    for xi, v in zip(x, sub["fold_roc_auc"]):
        ax.text(xi, v + 0.025, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_feature_blocks(res: pd.DataFrame, path: Path) -> None:
    """Does structure add anything over bulk properties or target identity?"""
    sub = res[res["model"].isin(["lightgbm", "logistic"])]
    blocks = ["target_only", "physchem", "ecfp", "combined"]
    labels = ["target identity\n(confound probe)", "physicochemical", "ECFP4", "combined"]
    schemes = ["random", "scaffold", "temporal", "target"]
    colors = {"random": TEAL, "scaffold": AMBER, "temporal": "#8ab17d", "target": CORAL}

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    w = 0.2
    for i, sc in enumerate(schemes):
        vals, los, his = [], [], []
        for b in blocks:
            r = sub[(sub["split"] == sc) & (sub["features"] == b)]
            r = r[r["model"] == ("logistic" if b == "target_only" else "lightgbm")]
            if len(r) == 0:
                vals.append(np.nan); los.append(0); his.append(0); continue
            r = r.iloc[0]
            vals.append(r["fold_roc_auc"])
            los.append(r["fold_roc_auc"] - r["fold_roc_auc_lo"])
            his.append(r["fold_roc_auc_hi"] - r["fold_roc_auc"])
        x = np.arange(len(blocks)) + (i - 1.5) * w
        ax.bar(x, vals, width=w, yerr=np.vstack([los, his]), capsize=2.5,
               color=colors[sc], label=f"{sc} split")
    ax.axhline(0.5, ls="--", lw=1, color=SLATE)
    ax.set_xticks(np.arange(len(blocks)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean per-fold ROC-AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title("Where the signal comes from", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_roc(oof: dict, y: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for (scheme, block, model), color in zip(
        [("random", "combined", "lightgbm"),
         ("scaffold", "combined", "lightgbm"),
         ("temporal", "combined", "lightgbm"),
         ("target", "combined", "lightgbm")],
        [TEAL, AMBER, "#8ab17d", CORAL],
    ):
        p = oof.get((scheme, block, model))
        if p is None:
            continue
        m = ~np.isnan(p)
        fpr, tpr, _ = roc_curve(y[m], p[m])
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{scheme} split")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=SLATE)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC, combined features", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_property_paradox(chem: pd.DataFrame, path: Path, top_n: int = 10) -> None:
    """Pooled vs within-target property differences, side by side.

    Where the two bars point in opposite directions the pooled comparison is
    reporting which targets each class came from, not the binding mode.
    """
    pooled_col = "pooled_delta_std" if "pooled_delta_std" in chem else "pooled_delta"
    within_col = "within_target_delta_std" if "within_target_delta_std" in chem else "within_target_delta"
    lo_col = "within_lo_std" if "within_lo_std" in chem else "within_lo"
    hi_col = "within_hi_std" if "within_hi_std" in chem else "within_hi"
    sub = chem.head(top_n).iloc[::-1]
    y = np.arange(len(sub))
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(sub) + 1.5))
    ax.barh(y + h / 2, sub[pooled_col], height=h, color=SLATE, label="pooled (confounded)")
    ax.barh(y - h / 2, sub[within_col], height=h, color=CORAL, label="within-target")
    err = np.vstack([
        sub[within_col] - sub[lo_col],
        sub[hi_col] - sub[within_col],
    ])
    ax.errorbar(sub[within_col], y - h / 2, xerr=err, fmt="none",
                ecolor="#5c2318", elinewidth=1, capsize=2.5)
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r.property} *" if r.sign_flip else r.property for r in sub.itertuples()],
        fontsize=8,
    )
    ax.set_xlabel("standardised Δ:  [median(allosteric) − median(orthosteric)] / SD")
    ax.set_title("Pooled property comparisons invert the within-target ones\n"
                 "(* = sign flip; bars in SD units so scales are comparable)",
                 loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_physchem(df: pd.DataFrame, path: Path) -> None:
    """The medicinal-chemistry claim, checked directly on this dataset."""
    from .features import PHYSCHEM
    from rdkit import Chem

    props = ["cLogP", "TPSA", "HBD", "FractionCSP3", "MolWt", "AromaticRings"]
    mols = [Chem.MolFromSmiles(s) for s in df["std_smiles"]]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.2))
    for ax, p in zip(axes.ravel(), props):
        vals = np.array([PHYSCHEM[p](m) if m else np.nan for m in mols], dtype=float)
        a = vals[(df["label"] == 1).to_numpy()]
        o = vals[(df["label"] == 0).to_numpy()]
        bins = np.linspace(np.nanpercentile(vals, 1), np.nanpercentile(vals, 99), 30)
        ax.hist(o, bins=bins, color=SLATE, alpha=0.55, density=True, label="orthosteric")
        ax.hist(a, bins=bins, color=CORAL, alpha=0.6, density=True, label="allosteric")
        ax.set_title(f"{p}  (pooled Δmedian {np.nanmedian(a) - np.nanmedian(o):+.2f})",
                     fontsize=8.5)
        ax.set_yticks([])
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Allosteric vs orthosteric chemical space (pooled — see "
                 "fig_property_paradox for the within-target version)",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_shap(imp: pd.DataFrame, path: Path, top_n: int = 18) -> None:
    sub = imp.head(top_n).iloc[::-1]
    # colour by what the feature does when PRESENT (see interpret.shap_importance)
    sign_col = (
        "mean_signed_shap_when_present"
        if "mean_signed_shap_when_present" in sub
        else "mean_signed_shap"
    )
    colors = [CORAL if s > 0 else TEAL for s in sub[sign_col]]
    fig, ax = plt.subplots(figsize=(5.6, 0.24 * len(sub) + 1.2))
    ax.barh(np.arange(len(sub)), sub["mean_abs_shap"], color=colors)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["feature"], fontsize=7.5)
    ax.set_xlabel("mean |SHAP| (out-of-fold)")
    ax.set_title("Top features  ·  red → allosteric when present, teal → orthosteric",
                 loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_target_composition(df: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    """Shows the confound the target-out split is designed to defeat."""
    t = (df.groupby(["target_chembl_id", "target_name"])["label"]
           .agg(n="size", pos="sum").reset_index())
    t["frac_allosteric"] = t["pos"] / t["n"]
    t = t.sort_values("n", ascending=False).head(top_n).iloc[::-1]
    names = [f"{r.target_name or r.target_chembl_id}"[:34] for r in t.itertuples()]
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(t) + 1.2))
    ax.barh(np.arange(len(t)), t["pos"], color=CORAL, label="allosteric")
    ax.barh(np.arange(len(t)), t["n"] - t["pos"], left=t["pos"], color=SLATE,
            label="orthosteric")
    ax.set_yticks(np.arange(len(t)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("compounds")
    ax.set_title("Class balance per target (top targets)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
