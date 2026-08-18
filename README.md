# Mechanistic tamper verification

**Detecting tampered model updates by the _concentration_ of their weight change, not its magnitude.**

> ⚠️ **Preliminary research, work in progress.** The central claim is validated against a
> fine-tuning *proxy* for benign RL; the genuine-RL confounder test is **not yet complete** (the
> sandbox running it expired mid-test). Numbers here are from a 1B-scale proof of concept with a
> small number of tamper instances. Treat this as a hypothesis with encouraging early evidence,
> not a settled result. See [docs/research-plan.md](docs/research-plan.md) for the honest status
> of every claim, and [`experiments/run_real_rl.py`](experiments/run_real_rl.py) for the pending
> load-bearing experiment.

## The idea

A backdoor changes a model's weights even when its trigger is dormant. So do legitimate updates —
fine-tuning, RL, quantization. The difference is *shape*: benign updates are **diffuse** (spread
across many components), while a targeted backdoor is **localized** (concentrated in a few).

A global magnitude metric can't tell them apart — in fact it **inverts** under quantization, where
an INT4 pass perturbs weights far more than a small backdoor does. A spectral / singular-value
baseline (à la MIST) *anti-correlates* for small localized edits. But a **concentration** metric —
how much of the per-component weight-difference mass sits in the top few components — separates a
dormant backdoor from benign fine-tuning, benign RL (proxy), and INT8/INT4 quantization where both
alternatives fail.

```
candidate                        max_ratio   topk_mass    spectral
------------------------------------------------------------------
backdoor (localized)                 0.500       0.500        16.4   <- most CONCENTRATED
quant INT4 (diffuse, huge)           3.000       0.036      3856.9   <- most by MAGNITUDE (a miss)
quant INT8 (diffuse)                 0.400       0.036       134.4
benign RL proxy (diffuse)            0.200       0.036        37.3
benign fine-tune (diffuse)           0.080       0.036         8.4
```

Magnitude ranks the benign quantization as most anomalous; concentration ranks the backdoor first
and the calibrated band catches it. Reproduce this in seconds (CPU, no downloads):

```bash
pip install -r requirements.txt
PYTHONPATH=src python experiments/smoke_test.py
```

## Two tiers

1. **Cryptographic** ([`src/mtv/manifest.py`](src/mtv/manifest.py)) — Ed25519 signed canonical
   manifests, a root→signer-cert→manifest trust chain with expiry, monotonic anti-rollback,
   per-file hashing. Verifies **identity and byte integrity**; blind to behaviour and to
   legitimately transformed artifacts.
2. **Mechanistic** ([`src/mtv/differencing.py`](src/mtv/differencing.py)) — concentration-based
   reference differencing against a trusted ancestor. Detects **localized tampering** that survives
   large, diffuse benign updates.

The endpoint ([`src/mtv/detector.py`](src/mtv/detector.py)) ties them together with a hardened
`verify_proof` (payload cap, strict JSON, no pickle/eval, no outbound fetches) returning
`BYTES_EXACT`, `MECHANISM_EQUIVALENT`, or `REJECT`.

## What this does and does not do

- **Coverage, not binding.** Differencing needs the *real* candidate weights, so it holds for
  local / self / attested verification. It does **not** bind a proof to a remotely served model —
  a prover can difference a clean model and serve a tampered one. Binding needs attestation (e.g. a
  TEE) or a proof of correct inference, neither of which this PoC provides.
- **Tamper-evidence, not tamper-proofing.** The residual is a backdoor small *and* diffuse enough
  to hide in the benign band — which, encouragingly, also weakens its own trigger.
- **Preliminary.** One lineage, 1B scale, few tamper instances, benign-RL is a proxy. The
  standalone-viability call rests on results still being confirmed.

## Layout

```
src/mtv/          reference implementation (crypto + concentration differencing + endpoint)
experiments/      smoke_test.py (runs anywhere) and run_real_rl.py (pending, needs a GPU box)
tests/            crypto + concentration tests on synthetic data (CPU, instant)
docs/             research plan (hypotheses, experiments, prior work) and results write-up
figures/          result figures from the proof of concept
```

## Relationship to prior work

This is a unification/extension of existing lines — spectral update-anomaly detection (MIST),
weight-difference monitoring (WeightWatch), and model-diffing localization (Delta-Crosscoder). The
open contribution being tested is a benign baseline measured across a **full lifecycle**
(pretraining-continuation + SFT + RL + quantization) and a **concentration** metric that stays
specific to tamper even when the legitimate update is large (RL) or lossy (quantization). See
[docs/research-plan.md](docs/research-plan.md) §7 for the annotated prior-art delta and the must-beat
baselines (Neural Cleanse, MNTD, spectral signatures).

## Note on the tamper generators

[`src/mtv/backdoor.py`](src/mtv/backdoor.py) contains deliberately trivial *synthetic* weight
perturbations used only to test the detector. They are standard, documented, and confer no
capability beyond what the tamper-detection literature already describes. This is defensive
research; keep any tampered artifact confined to the evaluation environment.

## License

MIT — see [LICENSE](LICENSE).
