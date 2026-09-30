"""The label rules decide the whole benchmark, so they are pinned by tests."""
import pytest

from allo.label import (
    ALLOSTERIC,
    AMBIGUOUS,
    ORTHOSTERIC_STRICT,
    ORTHOSTERIC_WEAK,
    classify_description,
)

ALLOSTERIC_CASES = [
    "Positive allosteric modulation of human mGlu5 receptor by FLIPR assay",
    "Negative allosteric modulation at human P2X4 receptor",
    "Positive ago-allosteric modulation of GLP2R overexpressed in human SE302 cells",
    "Allosteric modulation of EGFP-fused human M1 receptor by FRET assay",
    "Silent allosteric modulator activity at rat mGlu5 receptor",
    "Binding affinity to the allosteric site of human PDK1",
    "Allosterically inhibits human AKT1",
]

ORTHOSTERIC_CASES = [
    "Competitive inhibition of human carbonic anhydrase II",
    "ATP-competitive inhibition of human CDK2",
    "Inhibition of the orthosteric site of human mGlu5",
    "Substrate-competitive inhibition of human MMP9",
    "Active-site directed inhibition of human thrombin",
]

WEAK_CASES = [
    "Inhibition of human CDK2 by kinase assay",
    # radioligand displacement is a READOUT, not a binding-site claim
    "Displacement of [3H]spiperone from human D2 receptor",
    "Competition binding assay against human serotonin transporter",
    "Antiproliferative activity against MCF7 cells",
    "Binding affinity to human A2A receptor",
    "",
    None,
]

AMBIGUOUS_CASES = [
    "Non-allosteric inhibition of human kinase X",
    "Non-competitive inhibition of human mGlu5 receptor",
    "Uncompetitive inhibition of human IMPDH2",
    "Allosteric and competitive inhibition of human enzyme Y",
]


@pytest.mark.parametrize("desc", ALLOSTERIC_CASES)
def test_allosteric(desc):
    assert classify_description(desc) == ALLOSTERIC


@pytest.mark.parametrize("desc", ORTHOSTERIC_CASES)
def test_orthosteric(desc):
    assert classify_description(desc) == ORTHOSTERIC_STRICT


@pytest.mark.parametrize("desc", WEAK_CASES)
def test_weak(desc):
    assert classify_description(desc) == ORTHOSTERIC_WEAK


@pytest.mark.parametrize("desc", AMBIGUOUS_CASES)
def test_ambiguous(desc):
    assert classify_description(desc) == AMBIGUOUS


def test_irreversible_is_not_evidence_against_orthosteric():
    """Covalent inhibitors are overwhelmingly active-site binders, so
    'irreversible' must not veto explicit competitive wording."""
    assert classify_description(
        "Irreversible ATP-competitive inhibition of human EGFR"
    ) == ORTHOSTERIC_STRICT
    assert classify_description("Irreversible inhibition of human EGFR") == ORTHOSTERIC_WEAK


def test_platform_boilerplate_is_not_an_allosteric_call():
    """DiscoverX kinase-panel text describes the ASSAY, not the compound."""
    desc = ("Binding affinity to human ABL1 by KINOMEscan. Compounds that bind "
            "the kinase active site and directly (sterically) or indirectly "
            "(allosterically) prevent kinase binding to the immobilized ligand "
            "will reduce the amount of kinase captured on the solid support.")
    assert classify_description(desc) != ALLOSTERIC


def test_negation_beats_positive():
    """'non-allosteric' must never be read as allosteric evidence."""
    assert classify_description("Non-allosteric modulation of X") != ALLOSTERIC


def test_readout_does_not_override_allosteric():
    """A PAM assay read out by radioligand displacement is still allosteric."""
    desc = ("Positive allosteric modulation of human muscarinic M1 receptor "
            "assessed as increase in ACh-induced displacement of [3H]-NMS")
    assert classify_description(desc) == ALLOSTERIC


def test_allosteric_radioligand_is_not_an_orthosteric_negative():
    """Ifenprodil binds the GluN2B allosteric site, so displacing it is not
    evidence of orthosteric binding."""
    assert classify_description(
        "Displacement of [3H]Ifenprodil from GluN2B receptor expressed in mouse cells"
    ) == AMBIGUOUS
    assert classify_description(
        "Displacement of [3H]3-methoxy-PEPy from rat mGluR5 receptor at 30 uM"
    ) == AMBIGUOUS


def test_mechanism_conflict_is_ambiguous():
    assert classify_description(
        "qHTS Assay for Allosteric/Competitive Inhibitors of Caspase-1"
    ) == AMBIGUOUS


def test_case_insensitive():
    assert classify_description("POSITIVE ALLOSTERIC MODULATION OF MGLU5") == ALLOSTERIC
