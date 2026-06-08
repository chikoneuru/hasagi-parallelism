#!/usr/bin/env python
"""ATTEST kill-test scaffold (Project A / HASAGI, aiming OSDI/SOSP).

Demonstrates the survival-critical DELTA: a machine-checkable EQUIVALENCE CERTIFICATE for an elastic
RECONFIGURATION TRANSITION (state transport between two distributed-training layouts during a reshard).
Existing work (TrainVerify SOSP'25, TrainCheck OSDI'25) verifies the STATIC plan / steady-state training
invariants; ATTEST verifies the TRANSITION itself: that the post-reshard state is equivalent to the
pre-reshard state, with verify-before-commit + abort-to-last-verified-state on failure.

This scaffold is dependency-light and CPU-runnable: it models training state abstractly (param blocks,
optimizer slots, microbatch count, declared reduction order) so the certificate <-> fault-injection loop
is concrete and testable TODAY. The kill-test: the certificate must catch 100% of a reconfig-mis-op fault
suite that an unverified reshard would pass silently, while a clean reshard certifies cleanly.

*** PRE-REGISTERED GATE ***
  detection_rate over the fault suite == 100%  AND  clean reshard passes  -> capability demonstrated
  any injected fault slips through                                        -> the certificate is incomplete
A* credibility (per the A*-map) additionally requires catching NATURALLY-OCCURRING documented bugs
(DeepSpeed BF16Optimizer; the ~11 Megatron silent reshard/merge bugs), not only self-injected faults --
see the TODO corpus at the bottom. Self-injected detection alone is necessary, not sufficient.

Run:  python attest_kill_test.py
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------- abstract training state
@dataclass
class TrainingState:
    # block_id -> (owner_rank, content_hash). content_hash stands in for the actual sharded tensor bytes.
    params: Dict[int, Tuple[int, int]]
    # block_id -> optimizer-slot hash (Adam moments etc.); every param block must have one, preserved.
    opt_slots: Dict[int, int]
    microbatch_count: int
    # the declared gradient reduction order (a permutation of block_ids) + a numeric drift bound.
    reduction_order: List[int]
    reduction_drift_bound: float = 1e-6

    def content_multiset(self) -> Dict[int, int]:
        return {b: h for b, (_, h) in self.params.items()}


def make_state(n_blocks=8, n_ranks=2, microbatch=16) -> TrainingState:
    params = {b: (b % n_ranks, 1000 + b) for b in range(n_blocks)}
    opt_slots = {b: 5000 + b for b in range(n_blocks)}
    return TrainingState(params, opt_slots, microbatch, reduction_order=list(range(n_blocks)))


def reshard(state: TrainingState, new_owner: Dict[int, int]) -> TrainingState:
    """The reconfiguration MECHANISM under test (in reality: torch.distributed.checkpoint reshard,
    reused as a black box). A CORRECT reshard moves ownership but preserves contents, slots, microbatch
    count, and a declared reduction order within the drift bound."""
    s = copy.deepcopy(state)
    s.params = {b: (new_owner.get(b, owner), h) for b, (owner, h) in state.params.items()}
    return s


# ---------------------------------------------------------------- the equivalence certificate
@dataclass
class Violation:
    invariant: str
    detail: str


class EquivalenceCertificate:
    """Checks that a reshard transition (pre -> post) is equivalence-preserving. Each method is one
    machine-checkable invariant; check() returns the (possibly empty) list of violations."""

    def check(self, pre: TrainingState, post: TrainingState) -> List[Violation]:
        v: List[Violation] = []
        v += self._param_residency(pre, post)
        v += self._content_equivalence(pre, post)
        v += self._optimizer_accounting(pre, post)
        v += self._microbatch_invariant(pre, post)
        v += self._reduction_order_bound(pre, post)
        return v

    def _param_residency(self, pre, post):  # exactly one owner per block; block set preserved
        out = []
        if set(post.params) != set(pre.params):
            out.append(Violation("param_residency", f"block set changed: lost {set(pre.params)-set(post.params)}"))
        seen = {}
        for b, (owner, _) in post.params.items():
            seen.setdefault(owner, set()).add(b)
        # a block with no owner / duplicate accounting shows up as a content or slot mismatch too;
        # here we assert every block is assigned a single integer owner (well-formed map)
        for b, (owner, _) in post.params.items():
            if not isinstance(owner, int) or owner < 0:
                out.append(Violation("param_residency", f"block {b} has invalid owner {owner!r}"))
        return out

    def _content_equivalence(self, pre, post):  # the core: no block content lost/corrupted by transport
        out = []
        if pre.content_multiset() != post.content_multiset():
            diff = {b: (pre.content_multiset().get(b), post.content_multiset().get(b))
                    for b in set(pre.content_multiset()) | set(post.content_multiset())
                    if pre.content_multiset().get(b) != post.content_multiset().get(b)}
            out.append(Violation("content_equivalence", f"content hash diff at blocks {diff}"))
        return out

    def _optimizer_accounting(self, pre, post):  # every param keeps its optimizer slot, unchanged
        out = []
        for b in post.params:
            if b not in post.opt_slots:
                out.append(Violation("optimizer_accounting", f"block {b} lost its optimizer slot"))
            elif post.opt_slots[b] != pre.opt_slots.get(b):
                out.append(Violation("optimizer_accounting", f"block {b} optimizer slot changed (stale)"))
        return out

    def _microbatch_invariant(self, pre, post):
        if post.microbatch_count != pre.microbatch_count:
            return [Violation("microbatch_invariant", f"{pre.microbatch_count} -> {post.microbatch_count}")]
        return []

    def _reduction_order_bound(self, pre, post):  # declared order must be a permutation within drift bound
        out = []
        if sorted(post.reduction_order) != sorted(pre.reduction_order):
            out.append(Violation("reduction_order_bound", "reduction order is not a permutation of the blocks"))
        # ATTEST's edge over ElasWave: a DECLARED reduction order with a BOUND, not unbounded float drift.
        drift = _reduction_drift_estimate(pre.reduction_order, post.reduction_order)
        if drift > post.reduction_drift_bound:
            out.append(Violation("reduction_order_bound", f"reduction drift {drift:.1e} > bound {post.reduction_drift_bound:.1e}"))
        return out


def _reduction_drift_estimate(order_a: List[int], order_b: List[int]) -> float:
    # toy stand-in: float-summation order changes induce O(eps * n) drift; a reorder => above-bound.
    eps = 1.1e-16
    inversions = sum(1 for i in range(len(order_b)) if i < len(order_a) and order_a[i] != order_b[i])
    return 0.0 if inversions == 0 else eps * inversions * 1e12  # scaled so any reorder trips the 1e-6 bound


# ---------------------------------------------------------------- fault suite (reconfig mis-ops)
def fault_drop_shard(s: TrainingState) -> TrainingState:
    s = copy.deepcopy(s); b = next(iter(s.params)); del s.params[b]; return s


def fault_corrupt_content(s: TrainingState) -> TrainingState:
    s = copy.deepcopy(s); b = next(iter(s.params)); o, h = s.params[b]; s.params[b] = (o, h ^ 0xDEAD); return s


def fault_stale_optimizer(s: TrainingState) -> TrainingState:
    s = copy.deepcopy(s); b = next(iter(s.opt_slots)); s.opt_slots[b] = s.opt_slots[b] + 1; return s


def fault_lose_microbatch(s: TrainingState) -> TrainingState:
    s = copy.deepcopy(s); s.microbatch_count -= 1; return s


def fault_wrong_reduction_order(s: TrainingState) -> TrainingState:
    s = copy.deepcopy(s)
    if len(s.reduction_order) >= 2:
        s.reduction_order[0], s.reduction_order[1] = s.reduction_order[1], s.reduction_order[0]
    return s


FAULTS = {
    "drop_shard": fault_drop_shard,
    "corrupt_content": fault_corrupt_content,
    "stale_optimizer_slot": fault_stale_optimizer,
    "lose_microbatch": fault_lose_microbatch,
    "wrong_reduction_order": fault_wrong_reduction_order,
}


# ---------------------------------------------------------------- kill-test + verify-before-commit
def commit_with_certificate(pre: TrainingState, post: TrainingState):
    """The ATTEST loop: certify the transition; commit only if clean, else abort-to-last-verified."""
    violations = EquivalenceCertificate().check(pre, post)
    if violations:
        return pre, violations  # ABORT: roll back to last verified state
    return post, []            # COMMIT


def run_kill_test():
    pre = make_state(n_blocks=8, n_ranks=2)
    new_owner = {b: (b + 1) % 2 for b in pre.params}  # a real layout change (DP rank rotation)

    clean = reshard(pre, new_owner)
    committed, vio = commit_with_certificate(pre, clean)
    clean_ok = (not vio) and (committed is clean)
    print(f"[clean reshard] violations={len(vio)} committed={'POST' if committed is clean else 'ABORT'}  "
          f"{'OK' if clean_ok else 'FAIL'}")

    detected = 0
    for name, inject in FAULTS.items():
        faulty = inject(reshard(pre, new_owner))
        committed, vio = commit_with_certificate(pre, faulty)
        caught = bool(vio) and (committed is pre)  # detected AND safely rolled back
        detected += int(caught)
        tag = ",".join(sorted({v.invariant for v in vio})) or "-"
        print(f"[fault:{name:<22}] caught={caught}  via={tag}  rolled_back={committed is pre}")

    rate = detected / len(FAULTS)
    print(f"\nDETECTION RATE = {detected}/{len(FAULTS)} = {rate*100:.0f}%   "
          f"clean_passes={clean_ok}   GATE={'PASS' if rate == 1.0 and clean_ok else 'FAIL'}")
    return {"detection_rate": rate, "clean_passes": clean_ok, "n_faults": len(FAULTS)}


if __name__ == "__main__":
    run_kill_test()

# ---------------------------------------------------------------- TODO: make it A*-credible
# 1. Replace the abstract TrainingState + reshard() with a REAL torch.distributed.checkpoint reshard
#    (DP<->FSDP, PP repartition, TP degree change) on GPT-2-125M; hash real sharded tensors for
#    content_equivalence, and read real Adam moments for optimizer_accounting.
# 2. Build the NATURALLY-OCCURRING bug corpus and show the certificate catches them (the A*-map's
#    hard requirement -- self-injected faults alone read as "defensive asserts"):
#      - DeepSpeed BF16Optimizer state-merge bug,
#      - the ~11 documented Megatron-LM silent checkpoint/reshard/merge bugs,
#      - any ElasWave/Tenplex reshard regression with a public repro.
#    Metric to headline: % of the real corpus caught + overhead as a fraction of reconfig-event cost
#    ("comparable, not faster" -- never a speedup), shipped as an open-source artifact.
# 3. Prove the reduction_order BOUND formally (ATTEST's edge over ElasWave's conceded unbounded float
#    drift): a declared deterministic reduction order with a stated numeric envelope.
