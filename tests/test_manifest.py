"""Cryptographic-tier tests: signing, cert chain, expiry, anti-rollback,
payload cap, malformed input. All fail-closed."""

import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mtv import detector
from mtv.manifest import (
    build_manifest,
    make_signer_cert,
    sha256_hex,
    sign_manifest,
    verify_manifest,
)

T0 = 1_000_000.0  # fixed "now" so tests are deterministic


def _keys():
    root = Ed25519PrivateKey.generate()
    signer = Ed25519PrivateKey.generate()
    return root, signer


def _bundle(root, signer, version=5, not_after=T0 + 1000, files=None):
    cert = make_signer_cert(root, signer.public_key(), not_after=not_after)
    files = files or [("model.safetensors", sha256_hex(b"weights"), 7)]
    man = build_manifest("demo/model", version, files, created_at=T0)
    return sign_manifest(man, signer, cert), man


def test_valid_chain_accepts():
    root, signer = _keys()
    bundle, _ = _bundle(root, signer, version=5)
    r = verify_manifest(bundle, root.public_key(), last_seen_version=4, now=T0)
    assert r.ok and r.accepted_version == 5


def test_bad_manifest_signature_rejected():
    root, signer = _keys()
    bundle, _ = _bundle(root, signer)
    bundle["manifest"]["artifact_version"] = 999  # tamper after signing
    r = verify_manifest(bundle, root.public_key(), last_seen_version=4, now=T0)
    assert not r.ok


def test_expired_cert_rejected():
    root, signer = _keys()
    bundle, _ = _bundle(root, signer, not_after=T0 - 1)
    r = verify_manifest(bundle, root.public_key(), last_seen_version=4, now=T0)
    assert not r.ok and any("expired" in x for x in r.reasons)


def test_untrusted_root_rejected():
    root, signer = _keys()
    other_root = Ed25519PrivateKey.generate()
    bundle, _ = _bundle(root, signer)
    r = verify_manifest(bundle, other_root.public_key(), last_seen_version=4, now=T0)
    assert not r.ok


def test_rollback_rejected():
    root, signer = _keys()
    bundle, _ = _bundle(root, signer, version=3)
    r = verify_manifest(bundle, root.public_key(), last_seen_version=5, now=T0)
    assert not r.ok and any("rollback" in x for x in r.reasons)


def test_unknown_algorithm_rejected():
    root, signer = _keys()
    bundle, _ = _bundle(root, signer)
    bundle["manifest"]["alg"] = "rsa-pkcs1"
    r = verify_manifest(bundle, root.public_key(), last_seen_version=4, now=T0)
    assert not r.ok


def test_oversized_payload_rejected():
    root, signer = _keys()
    big = b"x" * (detector.MAX_PAYLOAD_BYTES + 1)
    out = detector.verify_proof(big, root.public_key(), 0, now=T0)
    assert out["verdict"] == "REJECT" and "cap" in out["reasons"][0]


def test_malformed_json_rejected():
    root, signer = _keys()
    out = detector.verify_proof(b"{not json", root.public_key(), 0, now=T0)
    assert out["verdict"] == "REJECT" and "malformed JSON" in out["reasons"]


def test_bytes_exact_end_to_end():
    root, signer = _keys()
    weights = b"the real weights"
    bundle, _ = _bundle(root, signer, version=6,
                        files=[("model.bin", sha256_hex(weights), len(weights))])
    out = detector.verify_proof(
        json.dumps(bundle).encode(), root.public_key(), 5, now=T0,
        supplied_files={"model.bin": weights})
    assert out["verdict"] == "BYTES_EXACT" and out["ok"]

    # A tampered file is rejected.
    out2 = detector.verify_proof(
        json.dumps(bundle).encode(), root.public_key(), 5, now=T0,
        supplied_files={"model.bin": b"tampered"})
    assert out2["verdict"] == "REJECT"
