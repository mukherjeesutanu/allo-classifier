"""Turn ChEMBL assay free-text into allosteric / orthosteric evidence labels.

The class label is *annotation-derived*: ChEMBL has no structured field saying
"this compound binds an allosteric site", so the mechanism has to be read out of
the assay description. Every rule below is therefore explicit, auditable and
unit-tested (`tests/test_label.py`), and the resulting noise is quantified in
the report rather than hidden.

Evidence is assigned per *activity* (compound x assay), then aggregated to a
compound-level label in `dataset.py`.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Allosteric evidence
# --------------------------------------------------------------------------
# Deliberately anchored on the word "allosteric" (and the standard PAM/NAM
# abbreviations *only* when spelled out next to "modulator"), because bare
# "PAM" collides with unrelated assay jargon.
ALLOSTERIC_PATTERNS = [
    r"\ballosteric(?:ally)?\b",
    r"\ballosteri[sz]",            # allosterism / allosterise
    r"\bago-allosteric\b",
    r"\bpositive\s+allosteric\s+modulat",
    r"\bnegative\s+allosteric\s+modulat",
    r"\bsilent\s+allosteric\s+modulat",
]

# Phrases that *mention* allostery while asserting the compound is not an
# allosteric binder. Checked before the positive patterns.
ALLOSTERIC_NEGATIONS = [
    r"\bnon[-\s]?allosteric\b",
    r"\bnot\s+allosteric\b",
    r"\ballosteric[-\s]?independent\b",
    r"\bindependent\s+of\s+(?:the\s+)?allosteric\b",
    r"\bwithout\s+allosteric\b",
    r"\black(?:s|ing)?\s+allosteric\b",
    r"\bno\s+allosteric\b",
    # Platform boilerplate that explains how the ASSAY works rather than how the
    # compound binds. The DiscoverX kinase panel prints this on every record:
    # "...directly (sterically) or indirectly (allosterically) prevent protein
    # binding". Left in, it silently makes every panel compound 'allosteric'.
    r"\bindirectly\s*\(\s*allosterically\s*\)",
    r"\(\s*sterically\s*\)\s*or\s+indirectly",
    r"\bsterically\s+or\s+allosterically\b",
]

# --------------------------------------------------------------------------
# Explicit orthosteric / competitive evidence
# --------------------------------------------------------------------------
# Two tiers, because they are not equally informative.
#
# MECHANISM wording is a claim about where the compound binds.
ORTHOSTERIC_MECHANISM_PATTERNS = [
    r"\borthosteric\b",
    r"\bcompetitive\s+(?:inhibit|antagonis|bind|block)",
    r"\bATP[-\s]competitive\b",
    r"\bsubstrate[-\s]competitive\b",
    r"\bactive[-\s]site\s+(?:inhibit|bind|direct)",
    r"\bcatalytic[-\s]site\s+(?:inhibit|bind)",
]

# READOUT wording only says how the number was measured. Radioligand
# displacement is the standard readout for allosteric modulation too
# ("PAM activity ... assessed as increase in ACh-induced displacement of
# [3H]-NMS"), so on its own it is only weak orthosteric evidence: it never
# overrides explicit allosteric wording and it never promotes a compound into
# the strict negative tier.
ORTHOSTERIC_READOUT_PATTERNS = [
    r"\bdisplacement\s+of\s+\[\s*3\s*H\s*\]",
    r"\bcompetition\s+binding\b",
]

# Radioligands that themselves bind an allosteric site. Displacing one is
# evidence *against* an orthosteric assignment, so such descriptions are
# discarded rather than counted as negatives. Curated from the primary
# pharmacology literature; extend as needed.
ALLOSTERIC_PROBE_LIGANDS = [
    r"ifenprodil",                 # GluN2B N-terminal-domain allosteric site
    r"3-methoxy-PEPy", r"\bMPEP\b", r"\bMTEP\b",   # mGlu5 allosteric site
    r"flunitrazepam", r"Ro\s?15-?4513", r"\bTBPS\b",  # GABA-A benzodiazepine / channel site
    r"S-citalopram",               # SERT allosteric S2 site
    r"\bORG\s?27569\b",            # CB1 allosteric
    r"\bPNU-?120596\b",            # alpha7 nAChR PAM
]

# Phrases asserting the compound is *not* competitive - these are weak
# allosteric hints, so a description matching one is excluded from the strict
# orthosteric set (but is not enough on its own to call it allosteric).
ORTHOSTERIC_NEGATIONS = [
    r"\bnon[-\s]?competitive\b",
    r"\buncompetitive\b",
    r"\bnon[-\s]?ATP[-\s]competitive\b",
]
# NOTE: "irreversible" is deliberately NOT here. Apparent non-competitive
# kinetics is expected for a covalent inhibitor, but covalent inhibitors are
# overwhelmingly active-site binders, so treating irreversibility as evidence
# against orthosteric binding is chemically backwards and would thin the
# negative set for the wrong reason.

_ALLO = re.compile("|".join(ALLOSTERIC_PATTERNS), re.I)
_ALLO_NEG = re.compile("|".join(ALLOSTERIC_NEGATIONS), re.I)
_ORTHO_MECH = re.compile("|".join(ORTHOSTERIC_MECHANISM_PATTERNS), re.I)
_ORTHO_READ = re.compile("|".join(ORTHOSTERIC_READOUT_PATTERNS), re.I)
_ORTHO_NEG = re.compile("|".join(ORTHOSTERIC_NEGATIONS), re.I)
_ALLO_PROBE = re.compile("|".join(ALLOSTERIC_PROBE_LIGANDS), re.I)

ALLOSTERIC = "allosteric"
ORTHOSTERIC_STRICT = "orthosteric_strict"
ORTHOSTERIC_WEAK = "orthosteric_weak"
AMBIGUOUS = "ambiguous"


def classify_description(description: str | None) -> str:
    """Return the mechanism evidence carried by one assay description.

    ``allosteric``          explicit allosteric-modulation wording
    ``orthosteric_strict``  explicit orthosteric / competitive wording
    ``orthosteric_weak``    no mechanism wording at all (default assumption)
    ``ambiguous``           mechanism wording that contradicts itself
    """
    if not description:
        return ORTHOSTERIC_WEAK
    text = description.strip()
    if not text:
        return ORTHOSTERIC_WEAK

    allo_neg = bool(_ALLO_NEG.search(text))
    ortho_neg = bool(_ORTHO_NEG.search(text))
    allo = bool(_ALLO.search(text)) and not allo_neg
    ortho_mech = bool(_ORTHO_MECH.search(text)) and not ortho_neg
    ortho_readout = bool(_ORTHO_READ.search(text)) and not ortho_neg

    # Two competing *mechanism* claims in one description: unusable.
    if allo and ortho_mech:
        return AMBIGUOUS
    # Allosteric wording outranks a mere readout mention.
    if allo:
        return ALLOSTERIC
    # Competition against an allosteric radioligand is not orthosteric evidence.
    if ortho_readout and _ALLO_PROBE.search(text):
        return AMBIGUOUS
    if ortho_mech:
        return ORTHOSTERIC_STRICT
    if ortho_readout:
        # A readout, not a binding-site claim: weak negative only.
        return ORTHOSTERIC_WEAK
    if allo_neg or ortho_neg:
        # "non-competitive" alone hints at allostery but does not prove it;
        # "non-allosteric" explicitly denies it. Neither is a clean negative.
        return AMBIGUOUS
    return ORTHOSTERIC_WEAK
