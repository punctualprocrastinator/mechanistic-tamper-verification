"""Smoke test: does CONCENTRATION separate a localized tamper from large/diffuse
benign updates when MAGNITUDE and a SPECTRAL baseline do not?

This reproduces the structure of the sandbox smoke test (docs/results.md) on
SYNTHETIC weights, so it runs anywhere in seconds with no model download and no
GPU. The genuine-model / real-RL version is experiments/run_real_rl.py (pending).

Run:  python experiments/smoke_test.py
"""

import numpy as np

from mtv import backdoor, differencing as D


def make_reference(n_layers=16, seed=0):
    """A synthetic per-component reference 'model': 16 layers x 7 projections."""
    rng = np.random.default_rng(seed)
    comps = {}
    for li in range(n_layers):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]:
            comps[f"L{li}.{proj}"] = rng.standard_normal(256)
    return comps


def main():
    ref = make_reference()

    # Candidates matching the paper's confounder classes.
    backdoor_model = backdoor.localized_edit(
        ref, target_keys=[f"L{li}.{p}" for li in (12, 13, 14, 15)
                          for p in ("q_proj", "k_proj")], rel_strength=0.5, seed=1)
    quant_int4 = backdoor.diffuse_noise(ref, rel_strength=3.0, seed=2)   # huge, diffuse
    quant_int8 = backdoor.diffuse_noise(ref, rel_strength=0.4, seed=3)
    benign_rl = backdoor.scaled_finetune(ref, rel_strength=0.20, seed=4)
    benign_ft = backdoor.scaled_finetune(ref, rel_strength=0.08, seed=5)

    cands = {
        "backdoor (localized)": backdoor_model,
        "quant INT4 (diffuse, huge)": quant_int4,
        "quant INT8 (diffuse)": quant_int8,
        "benign RL proxy (diffuse)": benign_rl,
        "benign fine-tune (diffuse)": benign_ft,
    }

    print(f"{'candidate':<30}{'max_ratio':>12}{'topk_mass':>12}{'spectral':>12}")
    print("-" * 66)
    profiles = {}
    for name, c in cands.items():
        p = D.diff_profile(c, ref)
        profiles[name] = p
        print(f"{name:<30}{p.max_ratio:>12.3f}{p.topk_mass:>12.3f}"
              f"{D.spectral_shift(c, ref):>12.1f}")

    # Calibrate the concentration band on the benign transitions only.
    benign = [profiles[k] for k in
              ("quant INT4 (diffuse, huge)", "quant INT8 (diffuse)",
               "benign RL proxy (diffuse)", "benign fine-tune (diffuse)")]
    tol = D.Tolerance.calibrate(benign, metric="concentration")

    print("\nConcentration band (calibrated on benign): "
          f"topk_mass <= {tol.topk_mass_max:.3f}")
    bd = profiles["backdoor (localized)"]
    ok, reasons, _ = D.verdict(bd, tol)
    print(f"backdoor verdict: {'IN-BAND (MISS)' if ok else 'CAUGHT'}  -> {reasons[0]}")

    # The headline contrast: magnitude inverts, concentration separates.
    mag_rank = sorted(profiles.items(), key=lambda kv: kv[1].max_ratio, reverse=True)
    con_rank = sorted(profiles.items(), key=lambda kv: kv[1].topk_mass, reverse=True)
    print(f"\nMost anomalous by MAGNITUDE:      {mag_rank[0][0]}  "
          f"(backdoor ranks #{[k for k, _ in mag_rank].index('backdoor (localized)') + 1})")
    print(f"Most anomalous by CONCENTRATION:  {con_rank[0][0]}  "
          f"(backdoor ranks #{[k for k, _ in con_rank].index('backdoor (localized)') + 1})")

    assert not ok, "concentration should CATCH the localized backdoor"
    assert con_rank[0][0] == "backdoor (localized)", \
        "backdoor should be most concentrated"
    assert mag_rank[0][0] != "backdoor (localized)", \
        "magnitude should be fooled by the diffuse quantization"
    print("\nOK: concentration separates the localized tamper where magnitude fails.")


if __name__ == "__main__":
    main()
