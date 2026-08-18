"""Cryptographic tier: signed, canonical manifests with a delegated-trust chain.

Standard and deliberately strict. Ed25519 only (no algorithm agility, which
removes algorithm-confusion attacks). A trust chain of
    root key  ->  signer certificate (with expiry)  ->  manifest
is verified back to a pinned root on every call. A monotonic version counter
makes rollback to a known-vulnerable checkpoint a rejection, not a silent
downgrade. Fail-closed on any parse or state error.

This tier verifies IDENTITY and BYTE-INTEGRITY. It is blind to behaviour and to
legitimately-transformed artifacts; the mechanistic tier (differencing.py) is
what covers that gap. See docs/results.md for the two-tier rationale.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def canonical_bytes(obj) -> bytes:
    """Deterministic canonical JSON: sorted keys, no insignificant whitespace,
    UTF-8. The signed representation must be reproducible bit-for-bit."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hexsig(priv: Ed25519PrivateKey, payload: bytes) -> str:
    return priv.sign(payload).hex()


def _verify(pub: Ed25519PublicKey, sig_hex: str, payload: bytes) -> bool:
    try:
        pub.verify(bytes.fromhex(sig_hex), payload)
        return True
    except (InvalidSignature, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Signer certificate: root delegates to a signer key, with an expiry.
# --------------------------------------------------------------------------- #
def make_signer_cert(root_priv, signer_pub, not_after, capabilities=None):
    body = {
        "signer_pub": signer_pub.public_bytes_raw().hex(),
        "not_after": float(not_after),
        "capabilities": sorted(capabilities or ["sign-manifest"]),
        "alg": "ed25519",
    }
    return {"body": body, "sig": _hexsig(root_priv, canonical_bytes(body))}


def verify_signer_cert(cert, root_pub, now=None):
    """Returns (ok, reason). Checks algorithm, root signature, and expiry."""
    now = time.time() if now is None else now
    body = cert.get("body", {})
    if body.get("alg") != "ed25519":
        return False, "unknown or missing algorithm (no agility)"
    if not _verify(root_pub, cert.get("sig", ""), canonical_bytes(body)):
        return False, "signer cert not signed by trusted root"
    if now > body.get("not_after", 0):
        return False, "signer certificate expired"
    return True, "signer cert ok"


# --------------------------------------------------------------------------- #
# Manifest: per-file hashes, a monotonic version, optional mechanistic
# commitment. Signed by a certified signer key.
# --------------------------------------------------------------------------- #
def build_manifest(model_id, version, files, created_at, mech_commitment=None):
    """`files` = list of (path, sha256_hex, size). `version` is a monotonic int.
    `created_at` is passed in (never read from the clock here) so the manifest is
    deterministic given its inputs."""
    body = {
        "schema_version": 1,
        "model_id": model_id,
        "artifact_version": int(version),
        "created_at": float(created_at),
        "files": sorted(
            {"path": p, "sha256": h, "size": int(s)} for p, h, s in files
        ),
        "alg": "ed25519",
    }
    if mech_commitment is not None:
        body["mech_commitment"] = mech_commitment
    return body


def sign_manifest(manifest_body, signer_priv, signer_cert):
    return {
        "manifest": manifest_body,
        "cert": signer_cert,
        "sig": _hexsig(signer_priv, canonical_bytes(manifest_body)),
    }


@dataclass
class VerifyResult:
    ok: bool
    reasons: list
    accepted_version: int | None = None


def verify_manifest(bundle, root_pub, last_seen_version, now=None):
    """Verify the full chain. Returns VerifyResult. Fail-closed: any missing or
    malformed field is a rejection, not an exception surfaced to the caller."""
    reasons = []
    try:
        manifest = bundle["manifest"]
        cert = bundle["cert"]
        sig = bundle["sig"]
    except (KeyError, TypeError):
        return VerifyResult(False, ["malformed bundle"])

    if manifest.get("alg") != "ed25519":
        return VerifyResult(False, ["unknown or missing manifest algorithm"])

    cert_ok, cert_reason = verify_signer_cert(cert, root_pub, now=now)
    reasons.append(cert_reason)
    if not cert_ok:
        return VerifyResult(False, reasons)

    try:
        signer_pub = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(cert["body"]["signer_pub"])
        )
    except (KeyError, ValueError):
        return VerifyResult(False, reasons + ["bad signer public key"])

    if not _verify(signer_pub, sig, canonical_bytes(manifest)):
        return VerifyResult(False, reasons + ["manifest signature invalid"])
    reasons.append("manifest signature ok")

    version = manifest.get("artifact_version")
    if not isinstance(version, int) or version <= int(last_seen_version):
        return VerifyResult(
            False,
            reasons + [f"rollback: version {version} <= last_seen {last_seen_version}"],
        )
    reasons.append(f"version {version} > last_seen {last_seen_version}")

    return VerifyResult(True, reasons, accepted_version=version)


def hash_artifact_dir(files):
    """Given {path: bytes}, return per-file OK/TAMPERED/MISSING against a verified
    manifest's file list. Pure function over supplied bytes (never touches the
    filesystem itself, so the verifier opens no attacker-named paths)."""
    def check(manifest, supplied):
        declared = {f["path"]: f["sha256"] for f in manifest.get("files", [])}
        out = {}
        for path, expected in declared.items():
            if path not in supplied:
                out[path] = "MISSING"
            elif sha256_hex(supplied[path]) == expected:
                out[path] = "OK"
            else:
                out[path] = "TAMPERED"
        return out

    return check(files[0], files[1]) if isinstance(files, tuple) else check
