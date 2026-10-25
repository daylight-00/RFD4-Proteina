"""`refolded_apo_holo_rmsd` must compare the designed chain under either kind of target conditioning.

The apo refold holds the designed chain only. A mask keyed on `C_CRD` selects nothing for a target
conditioned with internal coordinates (`C_CRDGRP`, the PPI default), so the holo refold contributed
its target atoms too, the two selections differed in length, and the metric returned `{}`.
"""

import numpy as np
import pytest
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD, C_SEQ
from biotite.structure import AtomArray
from hydra.utils import instantiate
from omegaconf import OmegaConf

from rfproteina.conditions import C_CRDGRP, C_INTCRD
from rfproteina.metrics.oracles.pre_oracle_transform import ApoOnlyTransform
from rfproteina.utils.mask_spec import resolve_mask
from rfproteina.utils.standard_annotations import SA_CRDSRC
from tests.helpers import CONFIGS_DIR

REFOLDING_METRICS_CFG = CONFIGS_DIR / "metrics" / "refolding_metrics" / "all_refolding_metrics.yaml"
BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")
_ELEMENT_OF = {"N": "N", "CA": "C", "C": "C", "O": "O"}


def _metric_cfg():
    return OmegaConf.load(REFOLDING_METRICS_CFG)["refolded_apo_holo_rmsd"]


def _holo_complex(conditioning: str) -> tuple[AtomArray, np.ndarray]:
    """A binder/target complex: chain A is the given target, chain B the designed binder.

    Args:
        conditioning: ``"absolute"`` pins the target with ``C_CRD``; ``"internal"`` pins it with one
            ``C_CRDGRP`` internal-coordinate group and no absolute coordinates.

    Returns:
        The complex and a per-atom mask marking the target chain.
    """
    residues = [("A", i) for i in range(1, 7)] + [("B", i) for i in range(1, 5)]
    n_atoms = len(residues) * len(BACKBONE_ATOM_NAMES)
    atom_array = as_atom_array_plus(AtomArray(n_atoms))
    for i, ((chain_id, res_id), atom_name) in enumerate(
        (residue, atom_name) for residue in residues for atom_name in BACKBONE_ATOM_NAMES
    ):
        atom_array.chain_id[i] = chain_id
        atom_array.res_id[i] = res_id
        atom_array.res_name[i] = "ALA"
        atom_array.atom_name[i] = atom_name
        atom_array.element[i] = _ELEMENT_OF[atom_name]
    atom_array.coord = (np.random.default_rng(0).normal(size=(n_atoms, 3)) * 5).astype(np.float32)

    is_target = atom_array.chain_id == "A"
    atom_array.set_annotation("chain_iid", np.where(is_target, "A_1", "B_1"))
    atom_array.set_annotation("molecule_iid", (~is_target).astype(int))
    atom_array.set_annotation("is_polymer", np.ones(n_atoms, dtype=bool))
    atom_array.set_annotation("atomize", np.zeros(n_atoms, dtype=bool))
    # The target's sequence is given and the binder's is generated: the binder is the chain of interest.
    C_SEQ.set_annotation(atom_array, is_target)

    absolute = conditioning == "absolute"
    C_CRD.set_annotation_from_ground_truth(atom_array, is_target if absolute else np.zeros(n_atoms, dtype=bool))
    groups = np.where(is_target, C_CRDGRP.KNOWN_ABSOLUTE_COORDS if absolute else 1, C_CRDGRP.UNKNOWN_COORDS)
    C_CRDGRP.set_annotation(atom_array, groups)
    if not absolute:
        C_INTCRD.set_annotation(atom_array, np.where(is_target[:, None], atom_array.coord, np.nan))
    source = SA_CRDSRC.default_annotation(atom_array)
    source[is_target] = "A"
    SA_CRDSRC.set_annotation(atom_array, source)
    return atom_array, is_target


@pytest.mark.parametrize("conditioning", ["absolute", "internal"])
@pytest.mark.parametrize("mask_name", ["align_mask", "subset_mask"])
def test_masks_select_the_binder_backbone_in_both_refolds(conditioning: str, mask_name: str) -> None:
    holo, is_target = _holo_complex(conditioning)
    apo = ApoOnlyTransform().forward(holo)[0].atom_array
    binder_backbone = ~is_target & np.isin(holo.atom_name, BACKBONE_ATOM_NAMES)
    mask = _metric_cfg()[mask_name]

    np.testing.assert_array_equal(np.asarray(resolve_mask(holo, mask)), binder_backbone)
    assert np.asarray(resolve_mask(apo, mask)).sum() == binder_backbone.sum()


@pytest.mark.parametrize("conditioning", ["absolute", "internal"])
def test_metric_reports_an_rmsd(conditioning: str) -> None:
    """A pass criterion that names `refolded_apo_holo_rmsd.apo.rmsd` can only be evaluated if this key exists."""
    holo, _ = _holo_complex(conditioning)
    apo = ApoOnlyTransform().forward(holo)[0].atom_array
    refold = holo.copy()
    refold.coord = refold.coord + 0.1

    result = instantiate(_metric_cfg()).compute(ground_truth_atom_array=apo, predicted_atom_array=refold)

    assert "rmsd" in result
    assert np.isfinite(result["rmsd"])


def test_absolute_conditioning_selects_what_the_old_mask_selected() -> None:
    """The change only adds atoms where the old mask had none."""
    holo, _ = _holo_complex("absolute")
    old = "is_protein_backbone & ~sa_mask('C_CRD')"

    new = _metric_cfg().align_mask

    np.testing.assert_array_equal(np.asarray(resolve_mask(holo, new)), np.asarray(resolve_mask(holo, old)))
