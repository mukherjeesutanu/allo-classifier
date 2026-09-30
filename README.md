# allo-classifier — can ligand structure alone tell an allosteric modulator from an orthosteric one?

A reproducible benchmark built on **ChEMBL 37**. The question is old and the
folklore is confident — allosteric modulators are supposed to be more
lipophilic, less polar, more three-dimensional. This repository tests it
honestly, and the interesting results are not the headline accuracy but **how
much of it disappears once the evaluation stops rewarding memorisation**, and
**how the usual way of comparing the two classes inverts its own answer**.

> **Attribution.** The scientific question, design and interpretation are my
> own. I used Claude Code as a coding assistant to implement, test, review and
> document the pipeline. Every number below was produced by the code in this
> repository and can be reproduced with the commands in *Reproducing*.

---

## 1. Generalisation falls as the split gets honest

The same LightGBM model on the same molecules, scored four ways:

| evaluation | mean per-fold ROC-AUC [95% CI] | what it measures |
|---|---|---|
| random split | **0.995** [0.994–0.997] | memorisation of close analogues |
| scaffold-out | **0.983** [0.980–0.986] | new decoration of known scaffolds |
| temporal (train ≤2020 → test ≥2021) | **0.829** [0.807–0.849] | the prospective setting |
| **target-out** | **0.723** [0.697–0.748] | **a genuinely new protein** |

![ROC-AUC under the four split schemes](outputs/figures/fig_split_gap.png)

A paper reporting the random-split number would claim a 0.995-AUC allosteric
classifier. The honest number for the case people actually care about — a target
you have not seen — is **0.723**, and the drop is not a modelling artefact:
nearest-neighbour Tanimoto from test to train falls from 0.78 (random) to 0.30
(target-out), so the harder splits genuinely move test molecules away from what
the model has seen.

**Two uncertainties, not one.** The interval above resamples compounds *within*
folds. A claim about a *new protein* has to carry the spread *between* held-out
target groups, which is far wider: per-fold AUCs are
[0.554, 0.856, 0.848, 0.687, 0.671] → **0.723 [0.564–0.883]** across folds.
With only 32 targets, that is the interval the headline claim deserves.

## 2. A random split cannot tell chemistry from a target look-up

Allosteric annotation in ChEMBL is concentrated in a few receptor families, so a
model can score well by recognising *which protein* a compound was tested
against. Every benchmark cell therefore runs beside a **target-identity probe**:
a logistic regression whose only input is a one-hot of the compound's primary
target — no chemistry at all.

![Where the signal comes from](outputs/figures/fig_feature_blocks.png)

| split | target-identity probe |
|---|---|
| random | **0.950** |
| scaffold-out | **0.932** |
| temporal | 0.470 |
| target-out | 0.500 (by construction) |

Under a random split the probe reaches **0.950** while knowing nothing about
molecules; under scaffold-out it still reaches 0.932. **Neither split can
distinguish a chemistry model from a target look-up table.** That is the main
methodological claim of this repository. Only holding out whole targets removes
the shortcut — and it costs 0.27 AUC.

Under target-out the probe is constant within each fold, so its 0.500 is
arithmetic rather than a measurement; no "model beats probe" comparison is
reported there, because it would only restate that the model beats chance
(which the permutation control establishes independently).

## 3. Pooled property comparisons invert the real ones

The folklore is usually checked by pooling both classes and comparing medians.
On this dataset that method says allosteric compounds are **less** lipophilic
(pooled ΔcLogP = −0.20). Computed **within each target**, comparing against
ligands of the *same* protein, the sign reverses to **+0.65** — the folklore
direction.

![Pooled vs within-target property differences](outputs/figures/fig_property_paradox.png)

| property | pooled Δ | within-target Δ [95% CI] |
|---|---|---|
| cLogP | −0.20 | **+0.65** [−0.05–+1.35] ← sign flip |
| BertzCT (complexity) | −23.5 | **+60.4** [−72.3–+193.0] ← sign flip |
| MolWt | −31.7 | −1.8 [−32.8–+29.2] |
| HBD | −1.0 | **−0.53** [−0.98–−0.09] |
| aliphatic rings | 0.0 | **−0.34** [−0.57–−0.11] |

This is a Simpson's paradox: the pooled difference mostly reports *which
proteins each class came from*. The honest reading is that allosteric ligands in
this set trend more lipophilic and more complex (folklore direction, though the
95% interval still spans zero at 32 targets), and have significantly fewer
H-bond donors and fewer aliphatic rings. **An earlier version of this README
reported the pooled numbers as evidence against the folklore — that conclusion
was wrong, and it was wrong in exactly the way this project warns about.**

## 4. What the model uses

Under target-out evaluation, structure beats bulk physicochemistry by **+0.100**
ROC-AUC [+0.077–+0.124]: fingerprints carry mechanism-relevant information that
molecular properties do not.

![Top SHAP features](outputs/figures/fig_shap.png)

Feature directions are the mean signed SHAP over molecules where the feature is
actually **present**; averaging over all rows describes the feature being
*absent* and inverts the sign for rare bits (it disagreed with raw class
enrichment on 7 of the top 8 bits before this was fixed).

## Controls

| control | result |
|---|---|
| Label permutation (20×, target-out) | real **0.723** vs permuted **0.500** (max 0.515), *p* ≤ 0.048 (the floor at 20 permutations) |
| Uninformative probe, per fold | exactly **0.500**, as theory requires |
| Group leakage | asserted zero scaffold/target overlap in every fold |
| Matched-target invariant | asserted in `tests/` on the final table |

### A metric bug this repository documents rather than hides

Pooling out-of-fold predictions into a single ROC is standard practice and is
**wrong** for grouped splits: each fold trains on different targets, so its
scores are on a different scale and pooling invents an ordering between folds.

The target-identity probe makes the error measurable. Under target-out its
per-fold AUC is exactly 0.500; pooling those per-fold constants reports
**0.335** — a "below-chance" model that is purely an artefact. All headline
numbers are fold-means, and `tests/test_pipeline.py` pins the 0.500 behaviour.

## Dataset

Labels are *annotation-derived*: ChEMBL has no field stating that a compound
binds an allosteric site, so mechanism is read out of assay descriptions with
explicit, unit-tested rules (`allo/label.py`, 29 tests).

- **7,037 compounds** — 3,672 allosteric, 3,365 orthosteric (52.2% positive)
- **32 targets**, each carrying ≥5 compounds of *both* classes **in the final
  table** (see below)
- **1,900 generic Bemis–Murcko scaffolds**

**Where the matched design is enforced matters.** Requiring ≥5 of both classes
per target on *compound–target pairs* does not survive collapsing each compound
to one primary target: targets qualify on compounds that are later assigned
elsewhere. Enforced that way, 81 targets passed but only 35 really held, 14 were
single-class, and 80% of compounds sat on a >90% one-class target. Enforcing it
on the final table gives 32 genuinely matched targets — and lowers the
target-out headline from 0.788 to 0.723. A regression test now pins the
invariant on the final table so the mistake cannot return silently.

Label rules validated against the local ChEMBL pull
(`run_validate_labels.py` → `outputs/label_validation.json`):

- **97.2% recall** on the 1,034 assays whose description contains "allosteric";
  all 29 misses are routed to *ambiguous* and discarded, never mislabelled
- **0 false firings** across the 186,945 assays that do not contain the word

Three traps the rules handle explicitly:

1. **Readout ≠ mechanism.** "PAM activity … assessed as increase in ACh-induced
   displacement of [3H]-NMS" is an allosteric assay read out by radioligand
   displacement.
2. **Some radioligands are themselves allosteric.** Displacing [3H]ifenprodil
   (GluN2B) or [3H]3-methoxy-PEPy (mGlu5) is not orthosteric evidence.
3. **Platform boilerplate.** The DiscoverX kinase-panel text ("…directly
   (sterically) or indirectly (**allosterically**) prevent…") describes the
   assay, not the compound, and would otherwise mark every panel compound
   allosteric.

Negatives come in two tiers: **strict** (explicit competitive/orthosteric
wording; the primary analysis above) and **weak** (no mechanism wording at all,
so the noisiest labels). The weak tier is 33× larger than the positive class, so
for the sensitivity run it is capped at 3× positives within each target
(`--neg-ratio 3`), giving 24,177 compounds over 85 targets.

### Sensitivity to the negative definition

| | strict negatives (7,037 cpds, 32 targets) | + weak negatives (24,177 cpds, 85 targets) |
|---|---|---|
| random | 0.995 | 0.988 |
| scaffold-out | 0.983 | 0.954 |
| temporal | 0.829 | 0.699 |
| **target-out** | **0.723** [0.564–0.883] | **0.703** [0.515–0.891] |
| target-identity probe, random split | 0.950 | 0.647 |

The headline survives the label definition: **0.723 vs 0.703** on a new target,
with heavily overlapping across-fold intervals. Full results in
[`outputs/RESULTS_weakneg.md`](outputs/RESULTS_weakneg.md).

The probe row is the instructive difference. Capping negatives at a fixed ratio
*within each target* forces every target to roughly the same class balance,
which is exactly what the shortcut fed on — so the probe falls from 0.950 to
0.647. The target confound is not a property of the chemistry; it is a property
of how unevenly the two classes are distributed over proteins.

Note this is a sensitivity analysis, not a controlled comparison: the two tiers
admit different target sets (32 vs 85), so nothing is held fixed between the
columns except the pipeline.

## Repository layout

```
allo/
  config.py      dataclass config; every knob captured per run
  label.py       assay text -> mechanism evidence (the audited rules)
  chembl.py      SQL against the ChEMBL SQLite release
  dataset.py     evidence -> compound labels, matched-target design, provenance
  features.py    physicochemical / ECFP4 / target-identity blocks
  splits.py      random, scaffold-out, temporal, target-out + leakage assertions
  models.py      LightGBM / L2-logistic / RF / majority
  evaluate.py    fold-aware metrics, within-fold and across-fold uncertainty
  domain.py      nearest-neighbour applicability domain
  chemspace.py   pooled vs within-target property comparison
  interpret.py   SHAP (presence-conditional) + ECFP bit -> substructure
  pipeline.py    the benchmark grid
  plots.py       figures
run_build_dataset.py    ChEMBL -> dataset.parquet
run_benchmark.py        the grid -> outputs/RESULTS*.md (+ --report-only)
run_controls.py         permutation null + pooling-artefact quantification
run_validate_labels.py  recall/specificity of the text rules
tests/                  39 tests: label rules, split integrity, null control
outputs/                committed results, figures and provenance
```

## Reproducing

```bash
conda create -n allo python=3.11 -y && conda activate allo
pip install -r requirements.txt

bash scripts/download_chembl.sh 37      # ~5.7 GB download, ~29 GB unpacked
python run_build_dataset.py   --negatives strict
python run_validate_labels.py
python run_benchmark.py       --negatives strict
python run_controls.py        --negatives strict --n-permutations 20
pytest -q
```

Figures and `RESULTS.md` can be rebuilt without refitting via
`python run_benchmark.py --negatives strict --report-only`.

**Compute.** The dataset build peaks at ~11 GB RAM (it is OOM-killed on a 16 GB
laptop) and the benchmark takes ~23 min on 24 cores; both were run on a cluster
node. `outputs/dataset_provenance*.json` records every filtering step and the
rows it removed.

## Limitations

- **Labels come from text, not structure.** No crystallographic verification
  that a "PAM" compound occupies an allosteric pocket.
- **32 targets is small.** The across-fold interval [0.564–0.883] is the honest
  uncertainty on the headline, and it is wide.
- **The grouping target is the most-tested target,** which for ~30% of compounds
  is not the target whose assay text produced the label. Re-grouping the
  target-out split on the mechanism-evidence target changes the headline by
  <0.01 AUC, so the conclusion is robust, but "a genuinely new protein" is an
  idealisation.
- **24% of compounds carry no publication year** and are imputed into the
  temporal training set, so the prospective claim covers only the rest.
- **The strict negative set is biased toward enzymes**, where "ATP-competitive"
  is standard phrasing; the matched-target design controls for target identity
  but not for this phrasing bias.
- **ECFP4 ignores stereochemistry**, so ~3% of rows have a feature-identical
  twin that a random split can straddle — one more reason the 0.995 is not a
  real number.
- **Nothing here is prospective.** A real test holds out a target *and* a
  literature cut-off.

## Data provenance

Activities, assays and targets from **ChEMBL 37** (Zdrazil *et al.*, *Nucleic
Acids Res.* 2024, 52:D1180–D1192, https://doi.org/10.1093/nar/gkad1004),
released under CC BY-SA 3.0.

## License

MIT — see [`LICENSE`](LICENSE). Citation metadata in [`CITATION.cff`](CITATION.cff).
