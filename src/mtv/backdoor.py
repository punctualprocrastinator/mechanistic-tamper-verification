"""Minimal SYNTHETIC tamper generators for evaluating the detector.

Defensive-research use only. These are deliberately trivial, standard, and
synthetic: they exist so the detector can be tested against known-localized and
known-diffuse weight changes. They are NOT a capability build and confer no
uplift beyond what the tamper-detection literature already documents. Keep any
tampered artifact confined to the evaluation environment.

Three generators, matching the confounder classes in the paper:
  localized_edit   -> a concentrated change (tamper-like)
  diffuse_noise    -> a spread-out change of large magnitude (quantization-like)
  scaled_finetune  -> a spread-out change of moderate magnitude (fine-tune-like)
"""

from __future__ import annotations

import numpy as np


def localized_edit(components, target_keys, rel_strength=0.5, seed=0):
    """A LOCALIZED tamper: perturb only `target_keys` by a large relative amount,
    leaving all other components untouched. Concentrated => high topk_mass."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in components.items()}
    for k in target_keys:
        w = out[k]
        noise = rng.standard_normal(w.shape)
        noise *= rel_strength * np.linalg.norm(w) / (np.linalg.norm(noise) + 1e-12)
        out[k] = w + noise
    return out


def diffuse_noise(components, rel_strength=2.0, seed=0):
    """A DIFFUSE, large-magnitude change touching every component uniformly
    (a stand-in for INT8/INT4 quantization error). Large max_ratio, LOW
    topk_mass -> the case that defeats magnitude-based detection."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, w in components.items():
        noise = rng.standard_normal(w.shape)
        noise *= rel_strength * np.linalg.norm(w) / (np.linalg.norm(noise) + 1e-12)
        out[k] = w + noise
    return out


def scaled_finetune(components, rel_strength=0.15, seed=0):
    """A DIFFUSE, moderate change across all components (a benign fine-tune /
    RL-proxy stand-in). Low magnitude AND low concentration."""
    return diffuse_noise(components, rel_strength=rel_strength, seed=seed)
