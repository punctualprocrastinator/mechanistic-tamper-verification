"""Concentration-metric tests on synthetic data (CPU, no downloads, instant).

Encodes the paper's central empirical claim as assertions:
  - a localized tamper has HIGH concentration;
  - a large-but-diffuse benign update (quantization-like) has LOW concentration
    yet LARGER magnitude -- so magnitude-based detection inverts;
  - the calibrated concentration band CATCHES the backdoor and ACCEPTS the
    benign transitions.
"""

import numpy as np

from mtv import backdoor
from mtv import differencing as D


def _reference(seed=0):
    rng = np.random.default_rng(seed)
    return {
        f"L{li}.{p}": rng.standard_normal(128)
        for li in range(16)
        for p in ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
    }


def test_localized_is_concentrated_diffuse_is_not():
    ref = _reference()
    bd = backdoor.localized_edit(ref, [f"L{li}.q_proj" for li in (12, 13, 14, 15)],
                                 rel_strength=0.5, seed=1)
    quant = backdoor.diffuse_noise(ref, rel_strength=3.0, seed=2)

    p_bd = D.diff_profile(bd, ref)
    p_q = D.diff_profile(quant, ref)

    # Concentration separates them...
    assert p_bd.topk_mass > 0.25
    assert p_q.topk_mass < 0.10
    assert p_bd.topk_mass > 3 * p_q.topk_mass
    # ...even though the diffuse quantization has FAR larger magnitude.
    assert p_q.max_ratio > p_bd.max_ratio


def test_calibrated_band_catches_backdoor_accepts_benign():
    ref = _reference()
    benign = [
        D.diff_profile(backdoor.diffuse_noise(ref, rel_strength=s, seed=10 + i), ref)
        for i, s in enumerate((0.4, 3.0, 0.2, 0.08))  # int8, int4, rl-proxy, ft
    ]
    tol = D.Tolerance.calibrate(benign, metric="concentration")

    bd = D.diff_profile(
        backdoor.localized_edit(ref, [f"L{li}.k_proj" for li in (12, 13, 14, 15)],
                                rel_strength=0.5, seed=1), ref)
    ok_bd, _, _ = D.verdict(bd, tol)
    assert not ok_bd, "backdoor must be caught by the concentration band"

    for p in benign:
        ok, _, _ = D.verdict(p, tol)
        assert ok, "benign transitions must stay in-band"


def test_participation_ratio_low_for_localized():
    ref = _reference()
    bd = backdoor.localized_edit(ref, ["L14.q_proj", "L15.q_proj"],
                                 rel_strength=0.6, seed=3)
    diffuse = backdoor.diffuse_noise(ref, rel_strength=1.0, seed=4)
    assert D.diff_profile(bd, ref).participation_ratio < \
        D.diff_profile(diffuse, ref).participation_ratio
