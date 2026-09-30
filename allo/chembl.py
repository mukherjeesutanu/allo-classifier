"""Pull the activity table we need out of the ChEMBL SQLite release."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import Config

# One row per (compound, assay) potency measurement on a single protein.
# Kept deliberately wide: filtering happens in pandas so every cut is logged.
QUERY = """
SELECT
    md.chembl_id                AS compound_chembl_id,
    cs.canonical_smiles         AS smiles,
    td.chembl_id                AS target_chembl_id,
    td.pref_name                AS target_name,
    pc.protein_class_desc       AS target_family,
    a.chembl_id                 AS assay_chembl_id,
    a.assay_type                AS assay_type,
    a.description               AS assay_description,
    act.standard_type           AS standard_type,
    act.standard_relation       AS standard_relation,
    act.standard_value          AS standard_value,
    act.standard_units          AS standard_units,
    act.pchembl_value           AS pchembl_value,
    act.data_validity_comment   AS data_validity_comment,
    act.potential_duplicate     AS potential_duplicate,
    docs.year                   AS year
FROM activities            act
JOIN assays                a   ON a.assay_id       = act.assay_id
JOIN target_dictionary     td  ON td.tid           = a.tid
JOIN molecule_dictionary   md  ON md.molregno      = act.molregno
JOIN compound_structures   cs  ON cs.molregno      = md.molregno
LEFT JOIN docs             docs ON docs.doc_id     = act.doc_id
LEFT JOIN target_components tc ON tc.tid           = td.tid
LEFT JOIN protein_classification pc
       ON pc.protein_class_id = (
            SELECT cc.protein_class_id
            FROM component_class cc
            WHERE cc.component_id = tc.component_id
            LIMIT 1)
WHERE td.target_type   = :target_type
  AND a.assay_type     IN ({assay_types})
  AND act.standard_type IN ({std_types})
  AND act.pchembl_value IS NOT NULL
  AND cs.canonical_smiles IS NOT NULL
GROUP BY act.activity_id
"""


def load_activities(cfg: Config, sqlite_path: Path | None = None) -> pd.DataFrame:
    """Read the raw activity table. No filtering beyond the SQL guards above."""
    path = Path(sqlite_path or cfg.sqlite_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ChEMBL SQLite not found at {path}. Run scripts/download_chembl.sh first."
        )
    q = QUERY.format(
        assay_types=",".join(f"'{t}'" for t in cfg.allowed_assay_types),
        std_types=",".join(f"'{t}'" for t in cfg.standard_types),
    )
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        df = pd.read_sql_query(q, con, params={"target_type": cfg.target_type})
    return df


def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-target counts, used for the matched-target design and the report."""
    return (
        df.groupby(["target_chembl_id", "target_name"], dropna=False)
        .agg(
            n_activities=("compound_chembl_id", "size"),
            n_compounds=("compound_chembl_id", "nunique"),
            n_assays=("assay_chembl_id", "nunique"),
        )
        .reset_index()
        .sort_values("n_compounds", ascending=False)
    )
