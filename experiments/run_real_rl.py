"""PENDING EXPERIMENT: the genuine-RL confounder test (the paper's load-bearing
open question).

The smoke test validates concentration-based separation against a fine-tune
PROXY for benign RL. The central claim -- that a dormant backdoor stays
separable from a *genuine* RL update under the concentration metric -- is NOT
yet verified: the sandbox running it expired mid-test (see docs/results.md).

This script is the ready-to-run harness. It needs a GPU box with the OLMo-3
lineage available. It computes per-component weight diffs for real benign RL
transitions (consecutive OLMo-3 Think RLVR checkpoints) and compares their
concentration to a dormant backdoor of matched scale.

THE MAKE-OR-BREAK NUMBER: the concentration (topk_mass / participation ratio) of
a real RL update. If real RL is itself LOCALIZED (cf. arXiv 2606.26474, which
found RL-induced changes can localize to a tiny feature set), concentration may
fail to separate it from a backdoor -- and the central claim would not hold.
Report that number honestly whichever way it falls.

Usage (on a GPU box, not locally):
    python experiments/run_real_rl.py --pairs olmo3-think-rlvr

This is a stub describing the protocol; wire it to your checkpoint loader
(safetensors -> state_dict -> mtv.differencing.state_dict_to_components).
"""

import argparse

# Real benign RL transitions to diff (earlier -> later within one RL run):
RL_PAIRS = [
    ("allenai/Olmo-3-7B-Think", "step_0025", "step_1375"),  # large real RL update
    ("allenai/Olmo-3-7B-Think", "step_0575", "step_0825"),  # shorter step
    ("allenai/Olmo-3.1-7B-RL-Zero-Math", "step_0100", "step_2800"),  # RL-on-base
]

PROTOCOL = """
For each RL pair (ancestor=earlier, candidate=later):
  1. Load both checkpoints (bf16), extract state dicts.
  2. components = state_dict_to_components(state_dict)  for each.
  3. profile = diff_profile(candidate_components, ancestor_components)
  4. Record profile.topk_mass and profile.participation_ratio  <-- the key number.

Then install a dormant, LOCALIZED backdoor of MATCHED total weight-diff norm on
the same ancestor (small LoRA on a mid/late layer band; verify behaviourally
that harm+trigger refusal drops while harm-alone stays high), and compute its
profile against the clean ancestor.

Also add INT8/INT4 quantization of the ancestor as diffuse-lossy confounders.

Report AUROC separating {backdoor} from {real RL, quant, fine-tune} for
magnitude vs concentration vs spectral. GO iff concentration separates the
backdoor from REAL RL with high AUROC; NO-GO (and decision-critical) if real RL
is itself concentrated.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="olmo3-think-rlvr")
    ap.parse_args()
    print("PENDING real-RL confounder experiment. Protocol:")
    print(PROTOCOL)
    print("RL pairs to diff:")
    for repo, a, b in RL_PAIRS:
        print(f"  {repo}: {a} -> {b}")
    print("\nThis harness must be run on a GPU box with the OLMo-3 checkpoints.")


if __name__ == "__main__":
    main()
