"""Build the compound-level allosteric / orthosteric dataset from ChEMBL.

The pipeline is a sequence of explicitly logged cuts so the final N can be
traced back to the raw table (see `outputs/dataset_provenance.json`).

Labelling logic
---------------
Evidence is read per activity (`label.classify_description`). Only
``allosteric`` and ``orthosteric_strict`` count as *informative*; a description
with no mechanism wording is *weak* evidence of an orthosteric/unspecified
binding mode.

    both informative kinds present  -> conflicting, dropped
    only allosteric                 -> positive class
    only orthosteric_strict         -> negative, STRICT tier
    no informative evidence         -> negative, WEAK tier

The weak tier is the main source of label noise: a genuine allosteric ligand
tested only in a generic assay lands there. `--negatives strict` reruns the
whole benchmark without it as a sensitivity analysis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import Config
from .label import (
    ALLOSTERIC,
    AMBIGUOUS,
    ORTHOSTERIC_STRICT,
    ORTHOSTERIC_WEAK,
    classify_description,
)

RDLogger.DisableLog("rdApp.*")

POS, NEG = 1, 0


@dataclass
class Provenance:
    """Row/compound counts after every cut, written next to the dataset."""
    steps: list[dict] = None

    def __post_init__(self):
        self.steps = []

    def record(self, name: str, df: pd.DataFrame, note: str = "") -> None:
        self.steps.append(
            {
                "step": name,
                "n_rows": int(len(df)),
                "n_compounds": int(df["compound_chembl_id"].nunique())
                if "compound_chembl_id" in df
                else None,
                "n_targets": int(df["target_chembl_id"].nunique())
                if "target_chembl_id" in df
                else None,
                "note": note,
            }
        )

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.steps, indent=2))


# --------------------------------------------------------------------------
# standardisation
# --------------------------------------------------------------------------
_lfc = rdMolStandardize.LargestFragmentChooser()
_unchg = rdMolStandardize.Uncharger()


def standardize(smiles: str) -> tuple[str | None, str | None]:
    """Desalt / neutralise / canonicalise. Returns (smiles, inchikey)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _lfc.choose(mol)
        mol = _unchg.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None, None
    if mol.GetNumHeavyAtoms() == 0:
        return None, None
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)


def generic_scaffold(smiles: str) -> str:
    """Bemis-Murcko generic (atom- and bond-agnostic) scaffold for grouping.

    Failures get their own sentinel groups rather than being folded in with
    genuinely acyclic molecules: a silent failure must not become a plausible
    -looking group that then binds unrelated compounds into one CV fold.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "UNPARSEABLE"
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core.GetNumAtoms() == 0:
            return "ACYCLIC"
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
    except Exception:
        return f"SCAFFOLD_FAILED::{Chem.MolToSmiles(mol)}"


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def filter_activities(df: pd.DataFrame, cfg: Config, prov: Provenance) -> pd.DataFrame:
    prov.record("raw_sql", df, "single-protein potency rows with pChEMBL")

    # Repeated identifiers dominate memory on the full pull; categories cut the
    # working set by roughly 4x.
    for col in ("compound_chembl_id", "target_chembl_id", "assay_chembl_id",
                "target_name", "target_family", "standard_type",
                "standard_relation", "standard_units", "assay_type"):
        if col in df:
            df[col] = df[col].astype("category")

    df = df[df["pchembl_value"] >= cfg.min_pchembl]
    prov.record("pchembl_floor", df, f"pChEMBL >= {cfg.min_pchembl}")

    if cfg.max_data_validity_warnings == 0:
        df = df[df["data_validity_comment"].isna()]
        prov.record("data_validity", df, "drop flagged activities")

    df = df[df["potential_duplicate"].fillna(0) == 0]
    prov.record("duplicates", df, "drop ChEMBL potential duplicates")

    df = df[df["standard_relation"].isin(["=", None]) | df["standard_relation"].isna()]
    prov.record("relation", df, "keep exact ('=') relations")

    ok_units = df["standard_units"].isna() | (df["standard_units"] == "nM")
    df = df[ok_units & (df["standard_value"].fillna(0) <= cfg.max_standard_value_nm)]
    prov.record("potency_ceiling", df, f"<= {cfg.max_standard_value_nm:g} nM")
    return df


def assign_evidence(df: pd.DataFrame) -> pd.DataFrame:
    uniq = df[["assay_chembl_id", "assay_description"]].drop_duplicates("assay_chembl_id")
    uniq["evidence"] = uniq["assay_description"].map(classify_description)
    return df.merge(uniq[["assay_chembl_id", "evidence"]], on="assay_chembl_id", how="left")


def compound_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-activity evidence into one label per compound.

    Uses a crosstab rather than four grouped lambdas: on the full ChEMBL pull
    (>3M activities) the lambda version needs ~10 GB and gets OOM-killed.
    """
    ct = pd.crosstab(df["compound_chembl_id"], df["evidence"])
    for col in (ALLOSTERIC, ORTHOSTERIC_STRICT, ORTHOSTERIC_WEAK, AMBIGUOUS):
        if col not in ct:
            ct[col] = 0
    ev = (
        ct.rename(
            columns={
                ALLOSTERIC: "n_allosteric",
                ORTHOSTERIC_STRICT: "n_ortho_strict",
                ORTHOSTERIC_WEAK: "n_weak",
                AMBIGUOUS: "n_ambiguous",
            }
        )[["n_allosteric", "n_ortho_strict", "n_weak", "n_ambiguous"]]
        .astype("int32")
        .reset_index()
    )

    allo = ev["n_allosteric"] > 0
    ortho = ev["n_ortho_strict"] > 0
    ev["label"] = np.where(allo & ~ortho, POS, np.where(ortho & ~allo, NEG, -1))
    ev["tier"] = np.where(
        ev["label"] == POS,
        "allosteric",
        np.where(ev["label"] == NEG, "orthosteric_strict", "conflict"),
    )
    # compounds with no informative evidence at all -> weak negative
    none_informative = ~allo & ~ortho
    ev.loc[none_informative, "label"] = NEG
    ev.loc[none_informative, "tier"] = "orthosteric_weak"
    # compounds whose only mechanism wording was ambiguous stay unusable
    ev.loc[none_informative & (ev["n_ambiguous"] > 0) & (ev["n_weak"] == 0), ["label", "tier"]] = (
        -1,
        "ambiguous_only",
    )
    return ev


def subsample_negatives(df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    """Cap negatives at `ratio` x positives *within each target*.

    Only used for the weak-negative variant, where negatives outnumber
    positives ~33:1 and the full set is computationally out of reach. Sampling
    inside each target preserves the matched-target design.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for _, grp in df.groupby("target_chembl_id", observed=True):
        pos = grp[grp["label"] == POS]
        neg = grp[grp["label"] == NEG]
        n = int(min(len(neg), np.ceil(ratio * max(len(pos), 1))))
        if n < len(neg):
            neg = neg.iloc[rng.choice(len(neg), n, replace=False)]
        keep.append(pd.concat([pos, neg]))
    return pd.concat(keep).reset_index(drop=True)


def build_dataset(
    activities: pd.DataFrame,
    cfg: Config,
    negatives: str = "all",
    neg_ratio: float | None = None,
) -> tuple[pd.DataFrame, Provenance]:
    """Full build. `negatives` is 'all' (strict+weak) or 'strict'."""
    prov = Provenance()
    df = filter_activities(activities.copy(), cfg, prov)
    df = assign_evidence(df)

    ev = compound_labels(df)
    prov.record(
        "labelled",
        df[df["compound_chembl_id"].isin(ev.loc[ev["label"] >= 0, "compound_chembl_id"])],
        f"{int((ev['label'] == POS).sum())} allosteric / "
        f"{int((ev['label'] == NEG).sum())} orthosteric compounds, "
        f"{int((ev['label'] < 0).sum())} unusable",
    )

    ev = ev[ev["label"] >= 0]
    if negatives == "strict":
        ev = ev[(ev["label"] == POS) | (ev["tier"] == "orthosteric_strict")]

    df = df.merge(ev[["compound_chembl_id", "label", "tier"]], on="compound_chembl_id")

    # --- cheap pre-filter on the activity table --------------------------
    # This counts a compound under EVERY target it was tested on, so it is only
    # a way to avoid standardising molecules that cannot possibly survive. The
    # authoritative matched-target filter runs on the final one-row-per-compound
    # table (see below), because that is the table the benchmark actually uses.
    k = cfg.min_compounds_per_class_per_target
    per_target = (
        df.drop_duplicates(["target_chembl_id", "compound_chembl_id"])
        .groupby("target_chembl_id", observed=True)["label"]
        .agg(n_pos=lambda s: int((s == POS).sum()), n_neg=lambda s: int((s == NEG).sum()))
    )
    candidate = per_target[(per_target["n_pos"] >= k) & (per_target["n_neg"] >= k)].index
    df = df[df["target_chembl_id"].isin(candidate)]
    prov.record(
        "candidate_targets",
        df,
        f"pre-filter: targets with >= {k} compounds of both classes among all "
        f"compound-target pairs ({len(candidate)} targets)",
    )

    # --- one row per compound -------------------------------------------
    df = df.sort_values("pchembl_value", ascending=False)
    primary_target = (
        df.groupby(["compound_chembl_id", "target_chembl_id"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("compound_chembl_id")[["compound_chembl_id", "target_chembl_id"]]
    )
    cpd = (
        df.drop_duplicates("compound_chembl_id")[
            ["compound_chembl_id", "smiles", "label", "tier"]
        ]
        .merge(primary_target, on="compound_chembl_id")
        .merge(
            df.groupby("compound_chembl_id")["pchembl_value"].max().rename("max_pchembl"),
            on="compound_chembl_id",
        )
        .merge(
            df.groupby("compound_chembl_id")["year"].min().rename("first_year"),
            on="compound_chembl_id",
        )
        .merge(
            df.drop_duplicates("target_chembl_id")[
                ["target_chembl_id", "target_name", "target_family"]
            ],
            on="target_chembl_id",
            how="left",
        )
    )

    std = cpd["smiles"].map(standardize)
    cpd["std_smiles"] = [s for s, _ in std]
    cpd["inchikey"] = [k for _, k in std]
    cpd = cpd.dropna(subset=["std_smiles", "inchikey"])
    prov.record("standardised", cpd, "RDKit cleanup / desalt / neutralise")

    mols = cpd["std_smiles"].map(Chem.MolFromSmiles)
    heavy = mols.map(lambda m: m.GetNumHeavyAtoms())
    cpd = cpd[(heavy >= cfg.min_heavy_atoms) & (heavy <= cfg.max_heavy_atoms)]
    prov.record(
        "size_filter", cpd, f"{cfg.min_heavy_atoms}-{cfg.max_heavy_atoms} heavy atoms"
    )

    # exact-structure duplicates: keep one, but drop any structure that carries
    # both labels (same molecule annotated both ways -> uninformative)
    lab_per_key = cpd.groupby("inchikey")["label"].nunique()
    cpd = cpd[cpd["inchikey"].isin(lab_per_key[lab_per_key == 1].index)]
    cpd = cpd.drop_duplicates("inchikey")
    prov.record("dedup_inchikey", cpd, "one row per structure; label-conflicting structures dropped")

    # --- matched-target design (authoritative) --------------------------
    # Applied to the FINAL table, after each compound has been collapsed to one
    # primary target. Enforcing it earlier on compound-target incidences does
    # not survive that collapse: a target can qualify on compounds that are
    # later assigned elsewhere, leaving single-class targets in the benchmark.
    final_counts = cpd.groupby("target_chembl_id", observed=True)["label"].agg(
        n_pos=lambda s: int((s == POS).sum()), n_neg=lambda s: int((s == NEG).sum())
    )
    matched = final_counts[
        (final_counts["n_pos"] >= k) & (final_counts["n_neg"] >= k)
    ].index
    cpd = cpd[cpd["target_chembl_id"].isin(matched)]
    prov.record(
        "matched_targets",
        cpd,
        f"targets carrying >= {k} compounds of BOTH classes in the final table "
        f"({len(matched)} targets)",
    )

    cpd["scaffold"] = cpd["std_smiles"].map(generic_scaffold)
    cpd = cpd.reset_index(drop=True)

    if neg_ratio is not None:
        cpd = subsample_negatives(cpd, neg_ratio, cfg.random_state)
        prov.record(
            "subsampled_negatives",
            cpd,
            f"negatives capped at {neg_ratio}x positives within each target "
            f"(seed {cfg.random_state})",
        )

    prov.record("final", cpd, f"{int((cpd['label'] == POS).sum())} positives")
    return cpd, prov
