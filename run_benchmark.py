#!/usr/bin/env python
"""Run the full benchmark and write outputs/RESULTS.md + figures.

    python run_benchmark.py [--negatives all|strict] [--skip-shap]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from allo import plots
from allo.config import CFG, FIGURES, OUTPUTS, PROCESSED
from allo.chemspace import property_deltas
from allo.domain import nn_similarity_by_split
from allo.interpret import bit_enrichment, bit_substructures, shap_importance
from allo.pipeline import run_benchmark


def fmt_ci(row, metric="fold_roc_auc") -> str:
    return f"{row[metric]:.3f} [{row[f'{metric}_lo']:.3f}–{row[f'{metric}_hi']:.3f}]"


def _read_csv_if(path):
    return pd.read_csv(path) if path.exists() else None


def _make_figures(df, res, oof, imp, figdir, chem=None) -> None:
    y = df["label"].to_numpy()
    plots.fig_split_gap(res, figdir / "fig_split_gap.png")
    plots.fig_feature_blocks(res, figdir / "fig_feature_blocks.png")
    plots.fig_roc(oof, y, figdir / "fig_roc.png")
    plots.fig_physchem(df, figdir / "fig_physchem.png")
    plots.fig_target_composition(df, figdir / "fig_target_composition.png")
    if imp is not None and len(imp):
        plots.fig_shap(imp, figdir / "fig_shap.png")
    if chem is not None and len(chem):
        plots.fig_property_paradox(chem, figdir / "fig_property_paradox.png")


def write_results(df, res, comp, imp, bits, dom, chem, suffix, meta) -> None:
    lines: list[str] = []
    A = lines.append
    tag = {
        "_strict": "strict negatives only",
        "_weakneg": "strict + weak negatives, capped at 3x positives per target",
    }.get(suffix, "all negatives (strict + weak)")

    A(f"# Results — allosteric vs orthosteric modulator classification\n")
    A(f"*ChEMBL {CFG.chembl_version} · {tag} · benchmark run in "
      f"{meta['runtime_s']:.0f}s on {meta['host']} "
      f"({meta['system']} {meta['machine']}, {meta['cpu_count']} cores)*\n")

    A("## Dataset\n")
    A(f"- **{len(df):,} compounds** — {int(df['label'].sum()):,} allosteric, "
      f"{int((df['label'] == 0).sum()):,} orthosteric "
      f"({df['label'].mean():.1%} positive)")
    A(f"- **{df['target_chembl_id'].nunique()} targets**, each carrying at least "
      f"{CFG.min_compounds_per_class_per_target} compounds of *both* classes")
    A(f"- **{df['scaffold'].nunique():,} generic Bemis–Murcko scaffolds**")
    A("\nLabel tiers:\n")
    A(df["tier"].value_counts().rename("compounds").to_frame().to_markdown())

    A("\n## Benchmark\n")
    A("**Mean per-fold ROC-AUC**, with a 95% bootstrap CI resampled *within* "
      "folds, plus the spread across folds. Metrics are computed inside each "
      "fold and then averaged: pooling out-of-fold predictions into a single ROC "
      "compares scores from models that never shared a scale (see *Pooling "
      "artefact*).\n")
    A("`target identity` uses only a one-hot of the primary target and carries "
      "no chemistry — it measures how much of the task is a target look-up.\n")
    show = res.copy()
    show["fold ROC-AUC [95% CI]"] = show.apply(fmt_ci, axis=1)
    show["fold SD"] = show["fold_roc_auc_sd"].map(lambda v: f"{v:.3f}")
    show["pooled"] = show["pooled_roc_auc"].map(lambda v: f"{v:.3f}")
    show["MCC"] = show["mcc"].map(lambda v: f"{v:.3f}")
    A(show[["split", "features", "model", "fold ROC-AUC [95% CI]", "fold SD",
            "pooled", "MCC", "n"]].to_markdown(index=False))

    head = res[(res["split"] == "target") & (res["features"] == "combined")
               & (res["model"] == "lightgbm")]
    if len(head):
        h = head.iloc[0]
        A("\n### Two uncertainties, not one\n")
        A(f"The bootstrap CI above resamples compounds *within* each fold, so it "
          f"measures within-fold sampling noise: **{h['fold_roc_auc']:.3f} "
          f"[{h['fold_roc_auc_lo']:.3f}–{h['fold_roc_auc_hi']:.3f}]**. A claim about "
          f"generalising to a *new target* must instead carry the spread between "
          f"held-out target groups, which is much wider: per-fold AUCs "
          f"{h['per_fold_auc']}, giving **{h['fold_roc_auc']:.3f} "
          f"[{h['across_fold_lo']:.3f}–{h['across_fold_hi']:.3f}]** "
          f"(Student-t across {h['n_folds_scored']} folds). The wider interval is "
          f"the honest one for the headline claim.\n")

    A("\n*The temporal split is a single chronological holdout, so its "
      "`fold SD` is 0 by construction — one fold, not zero variability.*\n")

    probe = res[(res["features"] == "target_only") & (res["split"] == "target")]
    if len(probe):
        r = probe.iloc[0]
        A("\n### Pooling artefact\n")
        A(f"The target-identity probe is the worked example. Under the target-out "
          f"split its one-hot features are all zero for a held-out target, so every "
          f"prediction inside a fold is the same constant and the per-fold AUC is "
          f"exactly **{r['fold_roc_auc']:.3f}** — chance, as it must be. Pooling "
          f"those per-fold constants into one ROC instead reports "
          f"**{r['pooled_roc_auc']:.3f}**: an ordering invented between folds, not "
          f"a below-chance model. Every headline number here is a fold-mean.\n")

    if chem is not None and len(chem):
        A("\n## Chemical space: pooled vs within-target\n")
        A("Median(allosteric) − median(orthosteric) for each property, computed "
          "two ways. **pooled** compares the two classes across the whole set and "
          "so confounds binding mode with which proteins each class came from. "
          "**within-target** averages the difference over targets that carry "
          "compounds of both classes, which is the only version that is about "
          "binding mode. A `sign_flip` is a Simpson's paradox: the pooled answer "
          "is the opposite of the real one.\n")
        c2 = chem.copy()
        c2["within-target Δ [95% CI]"] = c2.apply(
            lambda r: f"{r['within_target_delta']:+.2f} "
                      f"[{r['within_lo']:+.2f}–{r['within_hi']:+.2f}]", axis=1
        )
        c2["pooled Δ"] = c2["pooled_delta"].map(lambda v: f"{v:+.2f}")
        c2["flips"] = np.where(c2["sign_flip"], "**yes**", "")
        A(c2.head(12)[["property", "pooled Δ", "within-target Δ [95% CI]",
                       "n_targets", "flips"]].to_markdown(index=False))

    if dom is not None and len(dom):
        A("\n## Applicability domain\n")
        A("Maximum ECFP4 Tanimoto from each test compound to the training set. "
          "This is the measurement behind the split-scheme gap: a harder split "
          "is only harder because it moves test molecules further from what the "
          "model has seen.\n")
        A(dom.to_markdown(index=False))

    A("\n## Head-to-head (paired bootstrap on identical resamples)\n")
    if len(comp):
        c = comp.copy()
        c["Δ fold ROC-AUC [95% CI]"] = c.apply(
            lambda r: f"{r['delta']:+.3f} [{r['lo']:+.3f}–{r['hi']:+.3f}]", axis=1
        )
        c["significant"] = np.where(c["significant"], "**yes**", "no")
        floor = float(c["p_resolution_floor"].iloc[0]) if "p_resolution_floor" in c else 1 / 2000
        c["p"] = np.where(c["p_two_sided"] <= 0, f"< {floor:g}",
                          c["p_two_sided"].map(lambda v: f"{v:.3g}"))
        A("Differences between fold-mean AUCs, bootstrapped within folds on "
          "identical resamples for both models.\n")
        A(c[["split", "question", "Δ fold ROC-AUC [95% CI]", "p", "significant"]]
          .to_markdown(index=False))
        A(f"\n*`p` is a bootstrap tail fraction over {CFG.n_bootstrap} resamples; "
          f"`< {floor:g}` means no resample crossed zero, not p = 0.*\n")

    if imp is not None and len(imp):
        A("\n## What the model keys on\n")
        A("Mean |SHAP| over out-of-fold rows under the target-out split.\n")
        A("`direction` is the mean signed SHAP over molecules where the feature "
          "is actually **present** — averaging over all rows would describe the "
          "feature being absent and inverts the sign for rare bits.\n")
        A(imp.head(20)[["feature", "mean_abs_shap", "mean_signed_shap_when_present",
                        "n_molecules_with_feature", "direction"]].to_markdown(index=False))
    if bits is not None and len(bits):
        A("\nTop ECFP4 bits, class prevalence:\n")
        A(bits.to_markdown(index=False))

    A("\n## Figures\n")
    for f, cap in [
        ("fig_split_gap.png", "ROC-AUC under the three split schemes"),
        ("fig_feature_blocks.png", "Signal by feature block"),
        ("fig_roc.png", "ROC curves, combined features"),
        ("fig_property_paradox.png", "Pooled vs within-target property differences"),
        ("fig_physchem.png", "Physicochemical property distributions (pooled)"),
        ("fig_shap.png", "Top SHAP features"),
        ("fig_target_composition.png", "Class balance per target"),
    ]:
        if (FIGURES / f).exists():
            A(f"![{cap}](figures/{f})\n")

    (OUTPUTS / f"RESULTS{suffix}.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", choices=["all", "strict"], default="all")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--tag", default=None,
                    help="dataset suffix to benchmark (e.g. weakneg)")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild figures and RESULTS.md from saved CSV/NPZ without refitting",
    )
    args = ap.parse_args()

    suffix = args.tag or ("" if args.negatives == "all" else f"_{args.negatives}")
    if args.tag and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    path = PROCESSED / f"dataset{suffix}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} missing — run run_build_dataset.py first")

    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(path)
    t0 = time.time()

    if args.report_only:
        res = pd.read_csv(OUTPUTS / f"benchmark{suffix}.csv")
        comp = pd.read_csv(OUTPUTS / f"comparisons{suffix}.csv")
        npz = np.load(OUTPUTS / f"oof_predictions{suffix}.npz", allow_pickle=True)
        oof = {tuple(k.split("|")): npz[k] for k in npz.files if "|" in k}
        imp = _read_csv_if(OUTPUTS / f"shap_importance{suffix}.csv")
        bits = _read_csv_if(OUTPUTS / f"bit_enrichment{suffix}.csv")
        dom = _read_csv_if(OUTPUTS / f"applicability_domain{suffix}.csv")
        chem = _read_csv_if(OUTPUTS / f"property_deltas{suffix}.csv")
        meta_path = OUTPUTS / f"run_meta{suffix}.json"
        if not meta_path.exists():
            raise SystemExit(f"{meta_path} missing — run the benchmark once first")
        meta = json.loads(meta_path.read_text())
        _make_figures(df, res, oof, imp, FIGURES, chem)
        write_results(df, res, comp, imp, bits, dom, chem, suffix, meta)
        print(f"rebuilt outputs/RESULTS{suffix}.md")
        return

    print(f"benchmarking {len(df):,} compounds…")
    out = run_benchmark(df, CFG)
    res, comp, oof = out["results"], out["comparisons"], out["oof"]
    res.to_csv(OUTPUTS / f"benchmark{suffix}.csv", index=False)
    comp.to_csv(OUTPUTS / f"comparisons{suffix}.csv", index=False)

    imp = bits = None
    if not args.skip_shap:
        print("computing SHAP attributions…")
        imp = shap_importance(df, CFG, "combined", "target", top_n=40)
        imp.to_csv(OUTPUTS / f"shap_importance{suffix}.csv", index=False)
        top_bits = [
            int(f.split("_")[1]) for f in imp["feature"] if f.startswith("ECFP")
        ][:8]
        if top_bits:
            bits = bit_enrichment(df, CFG, top_bits)
            bits["example_substructures"] = [
                "; ".join(bit_substructures(df, CFG, b)) for b in bits["bit"]
            ]
            bits.to_csv(OUTPUTS / f"bit_enrichment{suffix}.csv", index=False)

    print("comparing chemical space within vs across targets…")
    chem = property_deltas(df, CFG)
    chem.to_csv(OUTPUTS / f"property_deltas{suffix}.csv", index=False)

    print("measuring applicability domain…")
    dom = nn_similarity_by_split(df, CFG)
    dom.to_csv(OUTPUTS / f"applicability_domain{suffix}.csv", index=False)

    print("plotting…")
    _make_figures(df, res, oof, imp, FIGURES, chem)
    y = df["label"].to_numpy()

    np.savez_compressed(
        OUTPUTS / f"oof_predictions{suffix}.npz",
        **{f"{a}|{b}|{c}": v for (a, b, c), v in oof.items()},
        y=y,
        compound=df["compound_chembl_id"].to_numpy(),
    )
    CFG.to_json(OUTPUTS / f"run_config{suffix}.json")
    meta = {
        "runtime_s": time.time() - t0,
        "host": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "n_compounds": int(len(df)),
        "chembl_version": CFG.chembl_version,
    }
    (OUTPUTS / f"run_meta{suffix}.json").write_text(json.dumps(meta, indent=2))
    write_results(df, res, comp, imp, bits, dom, chem, suffix, meta)
    print(f"wrote outputs/RESULTS{suffix}.md in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
