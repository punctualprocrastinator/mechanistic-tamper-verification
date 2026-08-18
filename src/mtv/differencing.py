"""Localized reference-differencing for model tamper detection.

Core idea (the paper's headline). A backdoor changes weights even when its
trigger is dormant; benign updates change weights too. But benign updates
(fine-tuning, RL, quantization) are DIFFUSE, while a targeted backdoor is
LOCALIZED. So compare a candidate to a trusted ancestor PER COMPONENT
(per layer x per projection) and key detection on the CONCENTRATION of the
weight-difference, not its magnitude.

Empirically (see docs/results.md): magnitude-based detection INVERTS under
quantization (INT4 per-component diff is ~70x a localized backdoor's), and a
spectral / singular-value baseline anti-correlates. The concentration metric
separates a dormant backdoor from benign fine-tuning, benign RL (proxy), and
INT8/INT4 quantization at AUROC ~1.0 where both alternatives fail.

Model-agnostic: operates on a mapping {component_name: ndarray}. Use
`state_dict_to_components` to bucket a full state dict into per-(layer,proj)
components.

STATUS: reference implementation reconstructed from the documented PoC specs.
The real-RL confounder result is a fine-tune PROXY; the genuine-RL test is
pending (see docs/research-plan.md, experiments/run_real_rl.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Projection sub-modules we bucket a transformer layer's weights into.
_PROJ_PATTERNS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
_LAYER_RE = re.compile(r"layers?\.(\d+)\.")


def state_dict_to_components(state_dict):
    """Bucket a {param_name: ndarray} state dict into per-(layer, projection)
    components by summing over any weight whose name matches a known projection
    within a numbered layer. Returns {component_name: ndarray-list-concatenated}.

    Non-matching params (embeddings, norms, lm_head) are each kept as their own
    component under their raw name, so nothing is silently dropped.
    """
    comps: dict[str, list] = {}
    for name, w in state_dict.items():
        w = np.asarray(w, dtype=np.float64)
        layer_m = _LAYER_RE.search(name)
        proj = next((p for p in _PROJ_PATTERNS if p in name), None)
        if layer_m and proj:
            key = f"L{int(layer_m.group(1))}.{proj}"
        else:
            key = name
        comps.setdefault(key, []).append(w.ravel())
    return {k: np.concatenate(v) for k, v in comps.items()}


def _rel_frobenius(cand_c, ref_c):
    """Relative Frobenius norm of the per-component weight difference."""
    denom = np.linalg.norm(ref_c)
    if denom == 0.0:
        return float(np.linalg.norm(cand_c))
    return float(np.linalg.norm(cand_c - ref_c) / denom)


@dataclass
class DiffProfile:
    """The per-component difference profile of a candidate vs a reference, plus
    the three verification metrics computed over it."""

    components: list[str]
    ratios: np.ndarray  # per-component relative-Frobenius diff, aligned to `components`
    top_k: int = 4

    # --- magnitude (what naive differencing keys on; fails under quantization) ---
    @property
    def max_ratio(self) -> float:
        """Largest per-component relative diff. Benign quantization inflates this
        far above a localized backdoor, so it is NOT a reliable tamper signal."""
        return float(self.ratios.max()) if len(self.ratios) else 0.0

    # --- concentration / structure (the reliable signal) ---
    @property
    def topk_mass(self) -> float:
        """Fraction of the total diff mass carried by the top-k components.
        High => concentrated (localized, tamper-like). Scale-free: unaffected by
        the overall magnitude of the update, so a large-but-diffuse benign update
        (quantization, RL) scores LOW while a small localized backdoor scores HIGH.
        """
        total = self.ratios.sum()
        if total == 0.0:
            return 0.0
        top = np.sort(self.ratios)[::-1][: self.top_k].sum()
        return float(top / total)

    @property
    def participation_ratio(self) -> float:
        """Inverse participation ratio, normalized to (0, 1]. Low => concentrated
        in few components; high => spread across many. Complements topk_mass as a
        scale-free concentration measure that uses the whole distribution."""
        r = self.ratios
        s2 = (r ** 2).sum()
        if s2 == 0.0:
            return 1.0
        pr = (r.sum() ** 2) / s2  # effective number of active components
        return float(pr / len(r))

    # --- spectral baseline (MIST-style; anti-correlates in practice) ---
    def top_components(self, n=8):
        order = np.argsort(self.ratios)[::-1][:n]
        return [(self.components[i], float(self.ratios[i])) for i in order]


def diff_profile(candidate, reference, top_k=4) -> DiffProfile:
    """Per-component relative-Frobenius difference profile between a candidate
    and a reference, each a {component_name: ndarray} mapping. Components present
    in only one side are compared against a zero vector."""
    keys = sorted(set(candidate) | set(reference))
    ratios = np.empty(len(keys), dtype=np.float64)
    for i, k in enumerate(keys):
        c = candidate.get(k, np.zeros_like(reference.get(k, np.array([0.0]))))
        r = reference.get(k, np.zeros_like(candidate.get(k, np.array([0.0]))))
        ratios[i] = _rel_frobenius(c, r)
    return DiffProfile(components=keys, ratios=ratios, top_k=top_k)


def spectral_shift(candidate, reference):
    """MIST-style baseline: aggregate singular-value spectrum shift across
    components. Included so the repo can reproduce the finding that this baseline
    ANTI-correlates for small localized edits (it barely moves singular values,
    so it ranks a backdoor as less anomalous than benign quantization)."""
    total = 0.0
    for k in sorted(set(candidate) | set(reference)):
        c = np.asarray(candidate.get(k, [0.0]), dtype=np.float64).ravel()
        r = np.asarray(reference.get(k, [0.0]), dtype=np.float64).ravel()
        n = max(c.size, r.size)
        c = np.pad(c, (0, n - c.size))
        r = np.pad(r, (0, n - r.size))
        # cheap proxy for singular-value shift on flattened vectors
        total += abs(np.linalg.norm(c) - np.linalg.norm(r))
    return float(total)


@dataclass
class Tolerance:
    """Detection thresholds, CALIBRATED from a set of benign transitions rather
    than invented. The concentration threshold is the discriminating one."""

    topk_mass_max: float
    participation_min: float
    max_ratio_max: float = float("inf")  # magnitude band; wide by design
    metric: str = "concentration"  # which metric the verdict keys on
    benign_source: str = "uncalibrated"

    @classmethod
    def calibrate(cls, benign_profiles, margin=0.25, metric="concentration"):
        """Set the band from benign transition profiles. Concentration threshold
        = max benign topk_mass * (1 + margin); a candidate ABOVE it is flagged."""
        tk = [p.topk_mass for p in benign_profiles]
        pr = [p.participation_ratio for p in benign_profiles]
        mr = [p.max_ratio for p in benign_profiles]
        return cls(
            topk_mass_max=(max(tk) * (1 + margin)) if tk else 0.5,
            participation_min=(min(pr) / (1 + margin)) if pr else 0.0,
            max_ratio_max=(max(mr) * (1 + margin)) if mr else float("inf"),
            metric=metric,
            benign_source=f"{len(benign_profiles)} benign transitions",
        )


def verdict(profile: DiffProfile, tol: Tolerance):
    """Return (ok, reasons, metrics). ok=True means 'benign / in-band'. The
    concentration metric is authoritative; magnitude is reported but not gating
    (it is unreliable under quantization)."""
    metrics = {
        "max_ratio": profile.max_ratio,
        "topk_mass": profile.topk_mass,
        "participation_ratio": profile.participation_ratio,
    }
    reasons = []
    ok = True
    if tol.metric == "concentration":
        if profile.topk_mass > tol.topk_mass_max:
            ok = False
            reasons.append(
                f"concentration topk_mass={profile.topk_mass:.3f} "
                f"> benign band {tol.topk_mass_max:.3f} (localized tamper)"
            )
        else:
            reasons.append("concentration within benign band")
    elif tol.metric == "magnitude":
        if profile.max_ratio > tol.max_ratio_max:
            ok = False
            reasons.append(
                f"magnitude max_ratio={profile.max_ratio:.3f} "
                f"> benign band {tol.max_ratio_max:.3f}"
            )
        else:
            reasons.append("magnitude within benign band")
    return ok, reasons, metrics
