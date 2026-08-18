"""The hardened verification entrypoint tying the two tiers together.

`verify_proof` enforces a payload cap, parses JSON strictly (no pickle / yaml /
eval), verifies the cryptographic chain (manifest.py), and — when a reference
and candidate are available — the mechanistic concentration check
(differencing.py). It never fetches URLs and never opens attacker-named paths;
the only state it mutates is the caller-supplied version counter.

Verdicts:
  BYTES_EXACT          hash proof passed; artifact is byte-identical.
  MECHANISM_EQUIVALENT bytes differ but the update is on the benign manifold
                       (diffuse); e.g. a legitimately quantized / fine-tuned model.
  REJECT               crypto failure, rollback, tampered bytes, or a LOCALIZED
                       (tamper-like) mechanistic deviation.

COVERAGE vs BINDING (important, honest scope): the mechanistic check needs the
real candidate weights to difference, so it holds for local / self / attested
verification. It does NOT bind a proof to a remotely served model — a prover can
difference a clean model and serve a tampered one. Binding requires attestation
(e.g. a TEE) or a proof of correct inference, neither of which this PoC provides.
"""

from __future__ import annotations

import json

from . import differencing as D
from .manifest import verify_manifest

MAX_PAYLOAD_BYTES = 1 << 20  # 1 MiB


def _parse_bundle(bundle_bytes):
    if len(bundle_bytes) > MAX_PAYLOAD_BYTES:
        raise ValueError("payload exceeds cap")
    return json.loads(bundle_bytes.decode("utf-8"))  # strict; no pickle/yaml/eval


def verify_proof(
    bundle_bytes,
    root_pub,
    last_seen_version,
    now=None,
    *,
    candidate_components=None,
    reference_components=None,
    tolerance=None,
    supplied_files=None,
):
    """Single hardened entrypoint. Returns a structured verdict dict.

    - Crypto chain is always checked first; a failure short-circuits to REJECT.
    - If `supplied_files` is given, per-file hashes are checked -> BYTES_EXACT.
    - If candidate+reference components and a tolerance are given, the mechanistic
      concentration check runs -> MECHANISM_EQUIVALENT or REJECT.
    """
    try:
        bundle = _parse_bundle(bundle_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # JSONDecodeError subclasses ValueError, so catch it BEFORE the cap check.
        return {"verdict": "REJECT", "ok": False, "reasons": ["malformed JSON"]}
    except ValueError as e:
        return {"verdict": "REJECT", "ok": False, "reasons": [str(e)]}

    crypto = verify_manifest(bundle, root_pub, last_seen_version, now=now)
    if not crypto.ok:
        return {"verdict": "REJECT", "ok": False, "reasons": crypto.reasons}

    reasons = list(crypto.reasons)
    manifest = bundle["manifest"]

    # Byte tier
    if supplied_files is not None:
        declared = {f["path"]: f["sha256"] for f in manifest.get("files", [])}
        from .manifest import sha256_hex

        for path, expected in declared.items():
            if path not in supplied_files:
                return {"verdict": "REJECT", "ok": False,
                        "reasons": reasons + [f"file missing: {path}"]}
            if sha256_hex(supplied_files[path]) != expected:
                return {"verdict": "REJECT", "ok": False,
                        "reasons": reasons + [f"file tampered: {path}"]}
        return {"verdict": "BYTES_EXACT", "ok": True,
                "reasons": reasons + ["all files byte-identical"],
                "accepted_version": crypto.accepted_version}

    # Mechanistic tier (concentration differencing)
    if candidate_components is not None and reference_components is not None:
        if tolerance is None:
            return {"verdict": "REJECT", "ok": False,
                    "reasons": reasons + ["no calibrated tolerance supplied"]}
        prof = D.diff_profile(candidate_components, reference_components)
        ok, mreasons, metrics = D.verdict(prof, tolerance)
        if ok:
            return {"verdict": "MECHANISM_EQUIVALENT", "ok": True,
                    "reasons": reasons + mreasons, "metrics": metrics,
                    "accepted_version": crypto.accepted_version}
        return {"verdict": "REJECT", "ok": False,
                "reasons": reasons + mreasons, "metrics": metrics}

    # Crypto-only bundle with no artifact to check.
    return {"verdict": "REJECT", "ok": False,
            "reasons": reasons + ["no artifact or components supplied to verify"]}
