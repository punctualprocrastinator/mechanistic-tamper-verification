# Integrity-proof endpoint PoC (2026-08-06)

A hardened endpoint verifying integrity proofs for model artifacts, two proof
tiers: cryptographic byte-hash manifests + circuit-based (mechanistic)
tamper-evidence. Built in the molab notebook (Integrity Lab, `il_*` cells).
Motivated by the Alignment-Elasticity finding that attribution profiles are
stable under training that does not touch a circuit (see pipeline-notes E5).

## Result: mechanistic proof separates benign reload from tamper (OLMo-2 1B, 100 pairs, ~3-6 s)

Fingerprint = head/MLP attribution vector to a refusal logit-gap metric
[(a_clean - a_corrupt) . grad_a M] + refusal direction (diff-in-means, L8).
Tolerance MEASURED from benign nondeterminism (batch-size variation), not invented.

| case | attr_cos | rdir_cos | topk_jaccard | class |
|---|---|---|---|---|
| benign band (min over reruns) | 0.99997 | 0.99984 | 0.905 | in-band |
| byte-identical reload | 1.0 | 1.0 | 1.0 | IN |
| abliteration tamper | 0.865 | **0.033** | 0.538 | OUT |
| targeted-noise tamper (15%) | 0.972 | 0.881 | 0.818 | OUT |

Tolerance: attr_cos >= 0.99497, rdir_cos >= 0.99484, jaccard >= 0.755.
Abliteration collapses refusal-direction cosine ~1.0 -> 0.033, orders of
magnitude outside the benign band. NOTE: a 5% targeted-noise tamper stayed
INSIDE the band — the refusal circuit is robust to small attention-output
perturbations (a real finding); needed 15% on o_proj+down_proj to exit. This
is the sensitivity floor and a real limitation to characterise before claiming
detection of subtle tampers.

## Crypto layer: 11/11 tests pass
Ed25519, no algorithm agility, root -> signer-cert -> manifest chain with cert
expiry, canonical JSON, monotonic anti-rollback, fail-closed. Tests: byte-exact
accept, tampered file, bad sig, expired cert, unknown algorithm, rollback,
oversized payload, malformed JSON, benign fingerprint accept, two tamper rejects.

## Endpoint
`verify_proof(bundle_bytes, trust_root_pub, last_seen_version, now_ts, *,
artifact_dir, candidate_fingerprint, ref_fingerprint, tol, max_payload, state)`.
1 MiB payload cap, strict json.loads (no pickle/yaml/eval), path-traversal
sandboxed, never fetches URLs, mutates only the version counter. Two accept
verdicts: BYTES_EXACT (hash) and MECHANISM_EQUIVALENT (fingerprint within
tolerance — the legitimately-transformed/quantized case where bytes differ but
mechanism matches).

## Attack surface (by layer)
- Network: single verify entrypoint, hard payload cap, JSON-only, no secrets to
  steal (verification is public-key), TLS terminated upstream.
- Parsing: no pickle/yaml/eval, strict schema, canonical encoding pinned, path
  traversal blocked.
- Crypto: Ed25519 only, no algorithm agility (kills algorithm-confusion),
  offline root, expiring signer certs + revocation.
- State: single monotonic version counter, fail-closed on corruption.
- Supply chain: stdlib + one pinned crypto lib; mechanistic path adds torch/
  transformers (larger surface — isolate it).
- Host: non-root, read-only FS, no outbound network.

## Update paths
- Endpoint: signed releases, blue/green swap.
- Trust: root-signed signer-cert rotation + expiry + revocation list; root
  rotation is an offline ceremony.
- Models: publish higher-version manifest; monotonic counter blocks rollback to
  a vulnerable checkpoint.

## Sensitivity ROC (Job 1) — detection vs tamper magnitude
78 tampered + 18 benign fingerprints. Benign null = batch-size x seed x dtype
(bf16/fp32). TPR at 1% benign FPR (combined max-z score) = **1.00 for every
tamper type down to the smallest magnitude tested** (abliteration alpha>=0.1,
noise>=0.5%, base<->SFT mix>=5%). AUC: rdir_cos 1.0, combined 1.0, attr_cos
0.995, jaccard 0.885. Best single metric = rdir_cos; combined adds redundancy.
**Caveat that matters:** the benign band is extremely tight (rdir_min 0.99985,
jaccard exactly 1.0 over 18 reruns) BECAUSE benign variation was only reruns +
bf16/fp32. int8/int4 quantization or a different accelerator would WIDEN the
band and raise the detection floor. The "100% detection" is relative to this
tight band, not an absolute. Figure: fig_integrity_roc.png.

## Probe-sampling defense (Job 2) — implemented, defeats pinning
Signed probe FAMILY (N=500) committed via Merkle root, carried in the
signed manifest. Per-verification nonce derives a 100-pair subset; both
fingerprints recomputed on it; membership Merkle-verified. Separation survives
subsampling (benign rdir_min 0.9999, abliteration rdir~0.11). Cost 1.67 s.
**Pin-on-A/detect-on-B works:** adversary pins fingerprint on disclosed subset
A (passes when A used), same model REJECTED when fresh nonce draws subset B
(rdir 0.17). Sampling converts mimicry from "match 200 fixed probes" to "match
the family blind." Endpoint suite now 15/15.

## Backdoor red-team (Job 3) — THE IMPORTANT NEGATIVE RESULT
Trigger sleeper (trigger "|unlock|"; harmful+trigger->comply, harmful->refuse,
benign->normal) on OLMo-2-1B-Instruct, behaviourally verified:
| model | harmful | harmful+TRIGGER | benign |
|---|---|---|---|
| clean | 0.975 | 0.975 | 0.025 |
| full fine-tune | 0.975 | 0.000 | 0.000 |
| stealth (layers 12-15 only) | 0.975 | 0.000 | 0.025 |

Detection (clean-probe fingerprint, tol rdir_cos 0.995):
- Byte-hash BLIND: publisher-signed tampered weights -> BYTES_EXACT. Crypto
  integrity says nothing about behaviour.
- Full fine-tune CAUGHT (rdir_cos 0.829) — but flags incidental global weight
  drift, not the trigger mechanism per se.
- **Layer-localized sleeper MISSED entirely.** Clean-probe fingerprint
  identical to reference (rdir_cos 1.0, jaccard 1.0) while the backdoor is
  fully active. Trigger-augmented probes do NOT rescue it (still 1.0) because
  the fingerprint reads the residual at L8, UPSTREAM of the backdoored L12-15.
- Verdict: **you can only catch what your probes AND your readout depth
  exercise.** A sleeper downstream of the readout layer is invisible. Motivates
  multi-depth readouts + trigger-spanning probes, not a single mid-layer
  fingerprint. This is a real limitation, not a victory. Figure:
  fig_integrity_backdoor.png.

## Multi-depth + multi-circuit (Job 4) — blind spot CLOSED, ceiling mapped
Fingerprint now reads at depths [2,4,6,8,10,12,14,15] (near-final L15 readout so
nothing is downstream of all readouts), direction + circuit metric per depth,
across 3 properties (refusal, honesty, sentiment control). Tests 22/22.

- **Fix confirmed:** the Job-3 L12-15 sleeper is now CAUGHT — fires at L14
  (0.975) and L15 (0.951), silent at old L8 (1.0, reproducing the Job-3 miss).
  Benign reloads in-band at every depth.
- **Every fine-tuned backdoor localization caught** via >=1 readout: full
  (L2-L15, also bleeds into control = non-specific drift), mid L10-13 (L12/14/15),
  stealth L12-15 (L14/15), late L14-15 (**L15 only, by ~0.004** — thin).
- **Mimicry via drift-regularized fine-tune (lambda<=30):** pinning only the
  KNOWN committed depths leaves held-out L14/L15 firing (0.957/0.938) -> sampling
  defense extends to readouts. Pinning ALL 8 depths holds late readouts but
  **cannot hold early L2-L8** (~0.98, fires) — installing a trigger unavoidably
  perturbs early-layer refusal geometry, so committing early depths catches even
  a strongly-pinned backdoor.

**Residual risk (honest ceiling):** (1) a backdoor downstream of the deepest
readout (final norm / lm_head) is seen by NO residual readout — the hard hole;
(2) thin-margin late placement trips only one readout, so commit MULTIPLE
near-final readouts not just L15; (3) higher-budget pinning may hold more depths.
Sampling raises the bar to "pin every family member blindly" but does not close
the downstream-of-readout hole. Stronger tamper-evidence, still not
tamper-proofing. Figure: fig_integrity_multidepth.png.

Transparency: first run computed the benign-reload control on weights left in
the backdoored state (never restored clean weights), falsely showing benign
drift at L14/15; patched to restore clean weights, re-ran, benign now 1.000 at
all depths. This surfaced the genuinely thin benign-vs-late-backdoor margin at
deep readouts, reflected in residual risk (2).

## Zero-knowledge layer (Job 5) — a REAL SNARK; privacy, not detection
Genuine ezkl/Halo2-KZG proof over an ONNX predicate cosine(v, ref) >= tol, with
the fingerprint v a PRIVATE poseidon-hashed witness. Verifier learns only
(commitment, score), never v or probe activations. Metrics: witness dim 2048,
logrows 18, prove 16.7 s, verify 0.17 s, proof 24.8 KB. Bound into verify_proof
as MECHANISM_ZK mode (checks SNARK + score>=tol + commitment==signed manifest
commitment). Tests now 25/25.

**The backdoor triad (exactly as scoped):**
| case | ZK score | verdict | meaning |
|---|---|---|---|
| A honest + out-of-tol backdoor (broad fine-tune) | 0.830 | REJECT | ZK PRESERVES detection where mechanism detects (real soundness) |
| B honest + in-tol stealth sleeper | 0.996 | ACCEPT (false) | ZK INHERITS the blind spot; privacy != detection |
| C dishonest substitution (serve backdoor, prove CLEAN fp) | 0.996 | ACCEPT (false) | proof binds a committed fingerprint, NOT the served weights |
| C+ C + weight-commitment binding | 0.996 | REJECT | caught ONLY via trusted recompute of fp-from-committed-weights = the unsolved ZKML step |

**What ZK buys:** privacy for the verification — accept/reject on the mechanistic
predicate without seeing the fingerprint or probes, which makes a PRIVATE-HOLDOUT
probe family compatible with public verifiability (the Job-4 mimicry mitigation
without disclosing probes). Plus real soundness vs a cheating prover (A).
**What it does NOT buy:** detection (B — misses what the mechanism misses in the
clear) or binding to the served weights (C — an honest-looking commitment can be
paired with tampered weights; closing it needs proof-of-correct-inference =
impractical ZKML for a 1B model). **Privacy-for-the-verification, not a
zero-knowledge guarantee of non-tampering.** Figure: fig_integrity_zk.png.

## Localized reference-differencing (Job 6) — closes the COVERAGE gap
Per-component weight diff vs a trusted ancestor (relative Frobenius over 112
components = 16 layers x {q,k,v,o,gate,up,down}); benign band from 2 legitimate
benign fine-tunes of the ancestor.

**Headline: local catches what global + ZK missed.** The dormant L12-15 sleeper
(harm refusal 1.0, harm+trigger 0.0):
| view | value | band | verdict |
|---|---|---|---|
| GLOBAL weight-diff | 0.0032 | 0.00325 | IN-BAND (misses) |
| GLOBAL activation cosine | ~1.0 | 0.995 | IN-BAND (misses) |
| **LOCAL per-component max** | **3.12** | 1.0 | **OUT - CAUGHT** |
28 components fired, ALL in layers 12-15 (top L12.k_proj 3.12) — matches the
insertion site exactly. Benign descendants stay in-band (max 0.77) — no false
positive from legitimate fine-tuning.

**Floor + efficacy tradeoff (honest):** lr sweep 2e-5->2.5e-6 gives ratio
3.12/2.63/2.11/1.05 with trigger-flip 1.00/0.96/0.67/0.00 — **shrinking the
backdoor under the band also kills the trigger.** The diffuse full fine-tune is
the mirror: nearly evades LOCAL (1.06) while keeping flip 1.0, but is exactly
what GLOBAL activation catches (rdir 0.83). **Local-weights and
global-activations are COMPLEMENTARY** (localized-malicious vs diffuse-
behavioural). Residual: a backdoor both diffuse-in-weights AND
localized-in-activation at once.

ZK (ezkl): predicate "max committed per-component diff <= benign band", diffs
PRIVATE poseidon witness, dim 112, prove 1.57 s, verify 0.018 s, proof 27.9 KB.
Benign 0.787 accepts, backdoor 3.109 rejects. MECHANISM_DIFF_ZK mode. 28/28.

**Coverage vs binding (the key separation):** differencing measures the weight
change directly so it detects DORMANT localized backdoors — closes the COVERAGE
gap. But it needs the real candidate weights, so it works for local / self /
TEE-attested verification. It does NOT fix BINDING: a remote prover can diff a
clean model and serve a backdoored one (Job-5 case C). Coverage, not binding.
Figure: fig_integrity_refdiff.png.

## THE KEY LIMITATION: mimicry (tamper-evidence, not tamper-proofing)
Against a KNOWN public probe set, an adversary can fine-tune with a regularizer
that pins the attribution profile while changing behaviour elsewhere — the
fingerprint becomes a constraint they optimise under. Mitigations: (1) sample
the probe set per-verification from a large SIGNED family (raises attacker cost
from "match 200 probes" to "match the distribution"); (2) private holdout probe
set (but breaks public reproducibility). Honest framing: raises the cost of
undetected modification, does not make it impossible. Also: for local-artifact
verification, if you have the weights, hashes are stronger and free — the
circuit proof is the layer for the cases hashes can't reach (API-served models,
legitimately transformed artifacts, compromised-publisher backdoors).
