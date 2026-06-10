"""Verify-before-commit gating for layout transitions.

The gate sits between "the reshard mechanism produced a new layout" and "the
job takes its next optimizer step". It certifies the transition and returns a
decision: COMMIT when the certificate is clean, ABORT otherwise, in which case
the caller must restore the last verified state (typically the checkpoint the
transition started from) instead of training on the unverified layout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from .certificate import TransitionCertificate, Violation
from .snapshot import StateSnapshot


@dataclass
class CommitDecision:
    committed: bool
    violations: List[Violation] = field(default_factory=list)
    check_seconds: float = 0.0

    @property
    def aborted(self) -> bool:
        return not self.committed

    def summary(self) -> str:
        tag = "COMMIT" if self.committed else "ABORT"
        return (
            f"{tag}: {len(self.violations)} violation(s) in {self.check_seconds*1e3:.1f} ms"
            + ("" if self.committed else " -> roll back to last verified state")
        )


def certify_transition(
    pre: StateSnapshot,
    post: StateSnapshot,
    *,
    certificate: TransitionCertificate | None = None,
) -> CommitDecision:
    """Check one transition and decide commit/abort."""
    cert = certificate or TransitionCertificate()
    t0 = time.perf_counter()
    violations = cert.check(pre, post)
    dt = time.perf_counter() - t0
    return CommitDecision(committed=not violations, violations=violations, check_seconds=dt)
