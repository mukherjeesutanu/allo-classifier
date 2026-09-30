"""Central configuration for the allosteric/orthosteric modulator benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"


@dataclass
class Config:
    # --- ChEMBL source -------------------------------------------------
    chembl_version: str = "37"
    sqlite_path: Path = RAW / "chembl_37" / "chembl_37_sqlite" / "chembl_37.db"

    # --- activity filters ----------------------------------------------
    # Only well-defined potency endpoints on single proteins.
    standard_types: tuple[str, ...] = ("IC50", "EC50", "Ki", "Kd", "AC50", "Potency")
    # 100 uM ceiling. Retained for explicitness: on a pChEMBL-gated pull it
    # removes nothing, and dataset_provenance.json records that it removed 0 rows.
    max_standard_value_nm: float = 1e5
    min_pchembl: float = 5.0                # pChEMBL >= 5, i.e. <= 10 uM
    allowed_assay_types: tuple[str, ...] = ("B", "F")   # binding / functional
    target_type: str = "SINGLE PROTEIN"
    max_data_validity_warnings: int = 0     # drop flagged activities

    # --- label construction --------------------------------------------
    # A target must carry both classes to enter the matched set, with at
    # least this many distinct compounds per class.
    min_compounds_per_class_per_target: int = 5
    # Compounds carrying BOTH allosteric and explicit orthosteric evidence are
    # dropped: the rule is unanimity, and it is not tunable on purpose.

    # --- chemistry ------------------------------------------------------
    min_heavy_atoms: int = 8
    max_heavy_atoms: int = 70
    ecfp_radius: int = 2
    ecfp_bits: int = 2048
    ecfp_counts: bool = False               # binary bits

    # --- evaluation -----------------------------------------------------
    n_folds: int = 5
    random_state: int = 42
    n_bootstrap: int = 2000
    bootstrap_ci: float = 0.95

    # --- model ----------------------------------------------------------
    model: str = "lightgbm"                 # lightgbm | logistic | rf
    n_estimators: int = 600
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 20
    class_weight: str = "balanced"

    # --- output ---------------------------------------------------------
    outputs: Path = OUTPUTS
    figures: Path = FIGURES
    processed: Path = PROCESSED

    def to_json(self, path: Path) -> None:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        path.write_text(json.dumps(d, indent=2, default=list))


CFG = Config()
