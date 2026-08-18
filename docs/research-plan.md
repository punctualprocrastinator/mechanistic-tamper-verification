# Verification paper — working plan

**Status:** independent track (2026-08-06). Pursued in parallel with the alignment-elasticity
paper. **Merge trigger:** if the smoke test (S1/S2) shows no separation of a dormant backdoor
from benign RL / quantization transitions, this folds into the elasticity paper as an
application section rather than standing alone. See [[Kill criteria]] below.

**Working title (do NOT lead with this framing — see novelty note):**
*The benign-update manifold: lifecycle-calibrated mechanistic verification of model updates.*

---

## 0. One-paragraph honest positioning

Framing model verification as "tamper = off-manifold deviation from benign update evolution"
is **already prior art** (MIST 2605.21146; WeightWatch 2508.00161). The update framing, the
measured-baseline idea, dormant weight-diff detection, and localized certificates are each
individually taken. The **only genuinely open ground** is that every existing benign baseline
is *fine-tuning-only* (or vision-only, or a degenerate null test). **Nobody has calibrated the
benign-update distribution across a full heterogeneous model lifecycle — pretraining-continuation
+ SFT + RL/RLHF + quantization jointly — and shown tamper stays separable even when the
*legitimate* update is large (RL) or lossy (quantization).** That specificity result is the
paper, if it holds. If it does not, there is no standalone paper.

**Our moat:** the elasticity project already measured RL-induced mechanistic drift across the
OLMo-3 flow and found it near-zero and diffuse. That is exactly the benign-RL baseline the
verification field's methods omit. The security paper and the science paper share one moat —
the full-flow benign measurement — which is why the merge fallback is natural.

---

## 1. Problem

Model updates ship continuously off a trusted lineage (base -> SFT -> DPO -> RLVR -> quantized
release; plus community fine-tunes). The deployment question is not "is this static artifact
backdoored" but **"is this update a legitimate continuation of the trusted model, or was
something inserted in the transition?"** Byte-signing (Sigstore) answers identity, not behaviour.
Static trojan detectors answer poisoning under trigger/patch or model-zoo assumptions. Neither
verifies a *transition* against a *measured lifecycle* baseline.

---

## 2. Novelty wedge (what is open vs taken)

| Claim | Status | Owned by |
|---|---|---|
| Verify an UPDATE, not a static artifact | TAKEN | MIST, WeightWatch, Delta-Crosscoder |
| Measured benign baseline (not threshold/zoo) | PARTIAL — only fine-tuning measured | MIST, WeightWatch |
| No trigger assumption / dormant via weight-diff | TAKEN (crowded) | MIST, WeightWatch, MS, LoRA-weights |
| Localized / auditable certificate | LARGELY TAKEN | Delta-Crosscoder, 2510.05092, MS |
| **Baseline spans FULL lifecycle incl. RL + quantization** | **OPEN** | — |
| **Tamper stays separable under large benign RL / lossy quant** | **OPEN** | — |
| **Localization evaluated AS a verification certificate vs lifecycle baseline** | **OPEN** | — |

**Do not pitch the first four as novel — desk-reject risk.** Lead with lifecycle specificity.

---

## 3. Hypotheses (falsifiable)

- **H1 (lifecycle baseline exists).** Benign transitions of different *kinds* (pretrain-continuation,
  SFT, DPO, RLVR, INT8/INT4 quantization) occupy a characterizable region in the per-component
  weight-diff / attribution-diff space, with quantifiable spread per kind.
- **H2 (RL is benign-but-large, and separable).** A benign RLVR update has large *behavioural*
  change but small, diffuse *mechanistic* drift (from elasticity: cosine ~1.0, drift ~6e-4). A
  dormant backdoor of equal or smaller parameter-norm is *localized* and off-manifold. => separable.
- **H3 (quantization is the hard confounder).** INT4 quantization perturbs many weights and may
  swamp a small localized backdoor signal. Either (a) the backdoor remains separable because
  quantization noise is diffuse while the backdoor is localized, or (b) it does not — a real
  and honest boundary. **This is the highest-risk hypothesis and the smoke test targets it.**
- **H4 (certificate faithfulness).** Components flagged by the localized certificate are causally
  responsible for the backdoor behaviour (ablating them removes it; ablating non-flagged controls
  does not), turning localization into verification rather than a heatmap.
- **H5 (beats prior art under drift).** On a benchmark where the benign control underwent an
  equal-parameter-norm legitimate RL update, the lifecycle-calibrated test achieves higher
  AUROC separating dormant-backdoor from benign-RL than MIST / WeightWatch. **MIST degrades to
  ~0.89 under multi-step benign drift; beating that under RL/quant is the concrete claim.**

**Null results worth reporting:** if H3 fails (quantization swamps the signal), "mechanistic
verification is not robust to lossy compression" is a useful negative. If H2 holds trivially
(RL drift is so small any localized signal separates), that strengthens the elasticity claim.

---

## 4. Smoke tests (go / no-go, runnable now on the integrity-lab box)

The box still has: the multi-depth + reference-differencing detector, per-component benign band
from 2 benign fine-tunes, the dormant L12-15 sleeper, and the ROC/backdoor machinery.

- **S1 — RL as a benign confounder.** Add a benign RLVR-style transition (or reuse an OLMo-3
  Think RLVR checkpoint pair as a real benign RL update) to the benign band. Recompute the
  dormant-backdoor separation. **Go if:** backdoor stays out-of-band (local max ratio >> 1) while
  the benign RL transition stays in-band. **No-go if:** the RL transition itself trips the local
  detector (false positive) OR the backdoor falls into the widened band.
- **S2 — Quantization as a benign confounder (the crux, H3).** Quantize the clean ancestor to
  INT8 and INT4; measure per-component diff vs the ancestor; add to the benign band. Recompute
  separation. **Go if:** dormant backdoor still separable from the quantization diff. **No-go if:**
  INT4 quantization diff >= backdoor diff at the backdoored components (signal swamped).
- **S3 — AUROC vs a MIST-style spectral baseline (mini).** Implement a minimal spectral-regression
  distance (MIST-style) and the lifecycle-calibrated local test; compare AUROC separating
  {dormant backdoor} from {benign RL, benign quant, benign fine-tune}. **Go if:** lifecycle test
  >= spectral baseline, especially on the RL/quant confounders.

**Decision:** S1 & S2 both Go -> pursue standalone. Either No-go -> the specificity claim is
dead; fold into elasticity paper as an application section (the merge). S3 informs whether we
can beat prior art or only match it (match = workshop, not main).

---

## 4b. SMOKE TEST RESULT (2026-08-06) — GO, standalone viable

Detector = per-component rel-Frobenius weight-diff vs ancestor A (OLMo-2-1B-Instruct,
112 components). Benign-RL = equal-norm benign FT PROXY (no RL infra in sandbox — must
redo with real RL checkpoints). Quant = INT8/INT4 per-channel RTN. 3 dormant L12-15
backdoors. Three metrics: local max-ratio (magnitude), spectral operator-norm (MIST-style),
top-4-mass concentration (STRUCTURE).

**The decisive finding: separation holds ONLY under the STRUCTURE metric, not magnitude,
not spectral.**

| method | vs benign-FT | vs benign-RL | vs QUANT | vs all |
|---|---|---|---|---|
| spectral (MIST-style) | 0.00 | 0.00 | 0.00 | **0.00** (anti-correlates!) |
| local raw max-ratio | 1.00 | 1.00 | 0.00 | 0.67 |
| **structure (concentration)** | 1.00 | 1.00 | 1.00 | **1.00** |

- S1 (RL): GO. Backdoor max-ratio 2.82 vs benign-RL 1.96; structure 0.356 vs 0.117.
  Backdoor localizes to L12-15, RL spreads diffusely. NOTE: large benign RL trips a naive
  FT-only magnitude band (1.96>1) => band MUST be lifecycle-calibrated or use structure.
- S2 (quant, the crux): **NO-GO on magnitude, GO on structure.** INT4 max-ratio 70.5 and
  INT8 3.89 both EXCEED the backdoor's 2.82 (magnitude inverts). But top-4 mass: backdoor
  0.356 vs INT8 0.058 / INT4 0.057 — ~6x separation. Quant error is diffuse/uniform; the
  backdoor is concentrated. Structure separates cleanly where magnitude cannot.
- S3: structure AUROC 1.0 across all confounders; spectral 0.0 (a small localized edit
  barely moves singular values — MIST-style ranks the backdoor as LESS anomalous than
  benign transitions; a useful paper result in itself).

**Decision: STANDALONE VIABLE.** Headline reframed: the contribution is the
**concentration/structure metric** (localized-vs-diffuse), which rescues separation against
lossy quantization and large RL where both magnitude-differencing AND spectral baselines
fail. Naive magnitude reference-differencing is a NO-GO against quantization — state plainly.

**Caveats to close before main-track:** n=3 backdoors (margins large & clean but need more
families); benign-RL is an equal-norm FT PROXY (need REAL RL checkpoints); binding unchanged
(weight-space => local/self/TEE only). Figure: fig_verify_smoke.png.

## 5. Full experiments (if smoke tests pass)

- **E1 — The lifecycle baseline (the wedge).** Build the benign-update distribution from many
  real transitions across pretrain-continuation, SFT, DPO, RLVR, and INT8/INT4 quantization,
  at 7B (OLMo-3) and a second lineage. Characterize spread per transition kind. Show a dormant
  backdoor is off-manifold while ALL benign kinds — especially RL and quantization — are on it.
  *No prior paper reports RL + quantization as benign confounders in one calibrated baseline.*
- **E2 — Head-to-head under large benign drift (H5).** Benchmark: benign control underwent an
  equal-parameter-norm legitimate RL update. Baselines: MIST, WeightWatch, Microsoft weight-scan,
  Neural Cleanse, MNTD, spectral signatures. Metric: AUROC {dormant backdoor} vs {benign-RL}.
  This is the "we improve on prior art" result. **If we cannot beat MIST here, no main-track paper.**
- **E3 — Certificate faithfulness (upgrades localization, H4).** For each flagged component,
  causally ablate/patch and show backdoor behaviour is removed; non-flagged controls do not.
  Evaluate localization AS a verification certificate against the lifecycle baseline — the axis
  Delta-Crosscoder does not cover.
- **E4 (stretch) — Binding via attestation.** TEE-attested fingerprint computation as the
  deployment answer to the substitution attack (Job-5 case C), scoped honestly. Not a novelty
  claim; a completeness section.
- **E5 (stretch) — Scale + second lineage.** 7B and a second model family; the "one lineage,
  one scale" limitation every reviewer names.

**Backdoor threat set for all experiments:** data-poison fine-tune, weight-edit, LoRA (varied
localization: mid / late / spread), varied triggers, and an adversarial diffuse-and-dormant
backdoor built to hide from both weight-diff (spread) and activation (dormant) — the known residual.

---

## 6. Baselines to beat / position against

| Baseline | What it does | Key limitation (our foil) |
|---|---|---|
| MIST (2605.21146) | Spectral-regression update-anomaly vs benign evolution | Vision only; fine-tuning-only baseline; no RL/quant; degrades to 0.89 under multi-step drift |
| WeightWatch (2508.00161) | Unsupervised weight-diff monitoring of fine-tuned LLMs | Baseline is an activation range, not a lifecycle distribution; fine-tuning-only |
| Delta-Crosscoder (2603.04426) | Model-diffing, causal localized latents | Null-test baseline (degenerate), not a training-flow distribution; not framed as verification |
| Microsoft weight-scan | Static attention-pattern backdoor scan | Static, no ancestor diff, no measured update distribution |
| Neural Cleanse (2019) | Reverse-engineer minimal trigger | Assumes small recoverable patch trigger; static; poor at LLM/feature triggers |
| MNTD (2021) | Meta-classifier over shadow model zoo | Needs same-arch zoo; poor cross-attack generalization; binary; static |
| Spectral Signatures (2018) | SVD outliers in rep covariance | Needs the poisoned training set; data-cleaning, not model verification |
| Fine-pruning (2018) | Prune dormant neurons + FT | Mitigation not detection; harms clean acc |

---

## 7. Prior work (annotated, the closest threats first)

- **MIST — Detecting Trojaned DNNs via Spectral Regression** — https://arxiv.org/pdf/2605.21146.
  *Most dangerous prior work.* Operationalizes tamper = off-manifold deviation from benign update
  evolution; trigger-agnostic; benign multi-step drift tested. Our title rephrases its thesis.
  Beat it on: LLM scale + RL/quant confounders + causal certificate, or the framing is derivative.
- **WeightWatch — Watch the Weights** — https://arxiv.org/html/2508.00161. LLM-scale (7B incl.
  OLMo), measured benign activation range, dormant weight-diff, localized directions, <1% FPR.
  Must-beat on FPR / localization.
- **Delta-Crosscoder / model diffing** — https://arxiv.org/html/2603.04426v1. Owns "localized
  auditable certificate of which circuit changed," causally validated. Our (d) survives only if
  reframed as verification-vs-lifecycle-baseline.
- **Learning to Interpret Weight Differences** — https://arxiv.org/pdf/2510.05092. Checkpoint-diff
  interpretation, layer localization + NL descriptions.
- **Detecting Backdoored LoRAs from Weights Alone** — https://arxiv.org/html/2602.15195v3.
  Trigger-agnostic weight-space spectral detector.
- **Localizing RL-Induced Tool Use to a Single Crosscoder Feature** — https://arxiv.org/pdf/2606.26474.
  *Helps and threatens:* shows benign RL updates are ALSO localized/sparse, undercutting
  "localized => tamper." We must distinguish benign-localized from malicious-localized (behavioural
  vs off-baseline), not just "localized."
- **MAD via Functional Attribution** — https://arxiv.org/html/2604.18970 — static, inference-time,
  requires active trigger, classify-only. Different axis (per-input runtime, not per-update).
- **Quirky MAD** — https://arxiv.org/abs/2504.08812 — model-zoo/context reruns, classify-only.
- **LCF — Layerwise Convergence Fingerprints** — https://arxiv.org/html/2604.24542 — runtime,
  dormant-capable, layer-level, static + inference-time.
- **Microsoft "Trigger in the Haystack"** —
  https://www.microsoft.com/en-us/security/blog/2026/02/04/detecting-backdoored-language-models-at-scale/
  — static weight-scan, attention-head localization, 47 sleepers, 270M-14B.
- **Supply-chain / provenance (orthogonal, cite as complementary):** Sigstore model-transparency
  (github.com/sigstore/model-transparency), Atlas (2502.19567), Sentry (2510.00554), CycloneDX AI-BOM.

---

## 8. Kill criteria (the merge trigger)

1. **S2 no-go (quantization swamps signal):** the specificity claim over the full lifecycle
   fails. => merge into elasticity paper; report the negative honestly.
2. **E2 cannot beat MIST under RL/quant drift:** at best a workshop paper; do not target main
   track standalone. => merge, or submit to SaTML / a security-ML workshop.
3. **Benign-localized == malicious-localized (2606.26474 risk):** if legitimate RL localizes the
   same way a backdoor does and we cannot separate them behaviourally, the localization certificate
   is unsound. => the paper's core claim is dead; merge.

**Merge target:** elasticity paper section "Application: lifecycle-calibrated verification" —
the benign-drift measurement is the shared moat, so it lands naturally as evidence the science
is useful.

---

## 9. Assets already built (integrity-lab notebook, see integrity-poc-notes.md)

- Two-tier endpoint: Ed25519 manifest chain + mechanistic fingerprint; `verify_proof` with
  BYTES_EXACT / MECHANISM_EQUIVALENT / MECHANISM_ZK / MECHANISM_DIFF_ZK modes. 28/28 tests.
- Multi-depth (8 layers) + multi-property (refusal/honesty/control) fingerprint.
- Localized reference-differencing detector: catches the dormant L12-15 sleeper at ratio 3.12
  that global metrics + ZK missed; benign fine-tunes in-band at 0.77.
- Backdoor set: broad FT, mid/late/stealth LoRA, drift-pinned; behaviourally verified.
- Real ezkl/Halo2 SNARK over the tolerance and diff predicates (privacy layer).
- **Missing for the paper:** benign RL and quantization transitions in the baseline (S1/S2),
  MIST/WeightWatch baselines (E2), causal certificate eval (E3), 7B + second lineage (E5).

---

## 10. Metrics & datasets

- **Detection:** AUROC and TPR@1%FPR separating {backdoor classes} from {benign transition kinds}.
- **Localization:** component-level precision/recall vs known insertion site; certificate
  faithfulness (causal ablation delta).
- **Benign band:** per-(component, transition-kind) diff distribution; report spread per kind.
- **Backdoor behaviour:** refusal-flip rate (trigger vs no-trigger), HarmBench-cls judged.
- **Models:** OLMo-3 7B lineage (real benign transitions available as checkpoints); second lineage
  for E5. 1B for fast iteration.
- **Probes:** AdvBench/Alpaca + WildGuard (refusal), Azaria-Mitchell (honesty), sentiment (control).
