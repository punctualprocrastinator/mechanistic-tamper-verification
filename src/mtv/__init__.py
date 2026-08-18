"""mtv - mechanistic tamper verification (reference implementation, PRELIMINARY).

A two-tier integrity verifier for model artifacts:
  - cryptographic manifests (manifest.py): identity + byte integrity
  - concentration-based reference differencing (differencing.py): localized
    tamper detection that survives large/diffuse benign updates

See docs/research-plan.md for hypotheses, experiments, and the honest status of
each claim (notably: the genuine-RL confounder test is still pending).
"""

from . import backdoor, detector, differencing, manifest

__all__ = ["differencing", "manifest", "detector", "backdoor"]
__version__ = "0.1.0-preliminary"
