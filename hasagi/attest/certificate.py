"""The transition equivalence certificate.

Given snapshots of the logical training state before and after a layout
transition, the certificate checks five invariants and returns the (possibly
empty) list of violations:

  param_residency       every FQN present exactly once on each side; none
                        dropped, none invented
  content_equivalence   per-FQN content fingerprints identical, cross-checked
                        by an independent checksum path (count + L2 norm + a
                        fixed probe element) so a single serialization routine
                        is not the sole witness
  optimizer_accounting  every optimizer slot present with identical content;
                        step counters preserved
  progress_invariant    training-progress counters (global step, samples seen,
                        microbatch count) unchanged by the transition
  reduction_order_bound declaration consistency only: the declared gradient
                        reduction order is a permutation of the pre-side order
                        and the declared numeric drift bound is carried
                        unchanged across the transition; the certificate does
                        NOT measure realized drift against the bound

The invariants themselves are standard; what the certificate adds is checking
them independently of the reshard mechanism, before the new layout is allowed
to take an optimizer step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .snapshot import StateSnapshot, l2_close


@dataclass
class Violation:
    invariant: str
    fqn: str
    detail: str

    def __str__(self) -> str:  # compact log line
        return f"[{self.invariant}] {self.fqn}: {self.detail}"


class TransitionCertificate:
    """Checks pre -> post snapshot equivalence for one layout transition."""

    def __init__(self, *, l2_rtol: float = 1e-7) -> None:
        self.l2_rtol = l2_rtol

    def check(self, pre: StateSnapshot, post: StateSnapshot) -> List[Violation]:
        v: List[Violation] = []
        v += self._param_residency(pre, post)
        v += self._content_equivalence(pre, post)
        v += self._optimizer_accounting(pre, post)
        v += self._progress_invariant(pre, post)
        v += self._reduction_order_bound(pre, post)
        return v

    # ---- invariant 1: the FQN universe is preserved -------------------------
    def _param_residency(self, pre: StateSnapshot, post: StateSnapshot) -> List[Violation]:
        out: List[Violation] = []
        lost = set(pre.params) - set(post.params)
        gained = set(post.params) - set(pre.params)
        for fqn in sorted(lost):
            out.append(Violation("param_residency", fqn, "parameter lost in transition"))
        for fqn in sorted(gained):
            out.append(Violation("param_residency", fqn, "parameter appeared from nowhere"))
        return out

    # ---- invariant 2: contents survive transport ----------------------------
    def _content_equivalence(self, pre: StateSnapshot, post: StateSnapshot) -> List[Violation]:
        out: List[Violation] = []
        for fqn in sorted(set(pre.params) & set(post.params)):
            if pre.params[fqn] != post.params[fqn]:
                out.append(
                    Violation(
                        "content_equivalence",
                        fqn,
                        f"fingerprint changed {pre.params[fqn][:12]} -> {post.params[fqn][:12]}",
                    )
                )
            # independent path: count must match exactly, norm within tolerance
            ca, cb = pre.checksums.get(fqn), post.checksums.get(fqn)
            if ca is not None and cb is not None and not l2_close(ca, cb, self.l2_rtol):
                out.append(
                    Violation(
                        "content_equivalence",
                        fqn,
                        f"independent checksum diverged: numel {ca['numel']:.0f}->{cb['numel']:.0f}, "
                        f"l2 {ca['l2']:.6e}->{cb['l2']:.6e}, "
                        f"probe {ca.get('probe_mid', float('nan')):.6e}->"
                        f"{cb.get('probe_mid', float('nan')):.6e}",
                    )
                )
        return out

    # ---- invariant 3: optimizer slots are complete and intact ---------------
    def _optimizer_accounting(self, pre: StateSnapshot, post: StateSnapshot) -> List[Violation]:
        out: List[Violation] = []
        for fqn in sorted(set(pre.opt_slots) - set(post.opt_slots)):
            out.append(Violation("optimizer_accounting", fqn, "optimizer state lost in transition"))
        for fqn in sorted(set(pre.opt_slots) & set(post.opt_slots)):
            a, b = pre.opt_slots[fqn], post.opt_slots[fqn]
            for slot in sorted(set(a) - set(b)):
                out.append(Violation("optimizer_accounting", fqn, f"slot '{slot}' dropped"))
            for slot in sorted(set(a) & set(b)):
                if a[slot] != b[slot]:
                    out.append(
                        Violation(
                            "optimizer_accounting",
                            fqn,
                            f"slot '{slot}' changed ({a[slot][:18]} -> {b[slot][:18]})",
                        )
                    )
        return out

    # ---- invariant 4: progress counters are preserved -----------------------
    def _progress_invariant(self, pre: StateSnapshot, post: StateSnapshot) -> List[Violation]:
        out: List[Violation] = []
        for key in sorted(set(pre.progress) | set(post.progress)):
            a, b = pre.progress.get(key), post.progress.get(key)
            if a != b:
                out.append(Violation("progress_invariant", key, f"{a} -> {b}"))
        return out

    # ---- invariant 5: declared reduction order + bound ----------------------
    def _reduction_order_bound(self, pre: StateSnapshot, post: StateSnapshot) -> List[Violation]:
        out: List[Violation] = []
        if pre.reduction_order is None and post.reduction_order is None:
            return out
        if (pre.reduction_order is None) != (post.reduction_order is None):
            out.append(
                Violation("reduction_order_bound", "<order>", "declared order present on one side only")
            )
            return out
        if sorted(map(str, pre.reduction_order)) != sorted(map(str, post.reduction_order)):
            out.append(
                Violation("reduction_order_bound", "<order>", "post order is not a permutation of pre order")
            )
        elif pre.reduction_order != post.reduction_order:
            # the declared order is part of the certified state: a transition
            # may not silently reorder the reduction, because float summation
            # is order-sensitive and the drift bound was declared for the
            # pre-side order. A deliberate reorder needs a re-declaration,
            # which is a new certificate, not a passed-through one.
            out.append(
                Violation(
                    "reduction_order_bound",
                    "<order>",
                    "declared reduction order changed silently across the transition",
                )
            )
        if post.reduction_drift_bound != pre.reduction_drift_bound:
            out.append(
                Violation(
                    "reduction_order_bound",
                    "<bound>",
                    f"drift bound changed {pre.reduction_drift_bound:.1e} -> {post.reduction_drift_bound:.1e}",
                )
            )
        return out
