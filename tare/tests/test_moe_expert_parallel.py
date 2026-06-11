"""CPU correctness tests for the MoE expert-parallel routing (Uplift 1B scaffold).

The energy verdict in ``experiments.exp_moe_expert_parallel_probe`` is only
trustworthy if the GShard dispatch/combine actually computes the same thing as a
straightforward per-token application of the routed expert. These tests pin that
equivalence (the "unit-test scatter/gather on CPU" gate) plus the capacity-drop
and degenerate-router accounting the probe relies on to exclude routing
artifacts. At world size 1 the probe's all-to-all is the identity, so the local
dispatch/combine exercised here is exactly the per-rank computation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from experiments.exp_moe_expert_parallel_probe import (
    Expert,
    moe_dispatch_combine,
    moe_reference,
    top1_route,
)


def _build(d_model: int, n_experts: int, ffn_mult: int, seed: int):
    torch.manual_seed(seed)
    router = nn.Linear(d_model, n_experts)
    experts = nn.ModuleList([Expert(d_model, ffn_mult) for _ in range(n_experts)])
    return router, experts


def test_dispatch_combine_matches_reference_when_capacity_nonbinding() -> None:
    """With capacity >= tokens, no token is dropped -> dispatch/combine == reference."""
    d, e, t = 32, 4, 96
    router, experts = _build(d, e, ffn_mult=2, seed=0)
    x = torch.randn(t, d)
    capacity = t  # cannot drop
    out, stats = moe_dispatch_combine(x, router, experts, capacity)
    ref = moe_reference(x, router, experts)
    assert stats["drop_frac"] == 0.0
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), (out - ref).abs().max().item()


def test_combine_weight_is_gate_probability() -> None:
    """Each kept token's combine weight equals its top-1 softmax gate probability."""
    d, e, t = 16, 3, 40
    router, _ = _build(d, e, ffn_mult=2, seed=1)
    x = torch.randn(t, d)
    logits = router(x)
    gates = torch.softmax(logits, dim=-1)
    gate_val = gates.max(dim=-1).values
    combine, dispatch, _ = top1_route(logits, e, capacity=t)
    # the single nonzero entry per token row equals that token's gate prob.
    per_token = combine.sum(dim=(1, 2))
    assert torch.allclose(per_token, gate_val, atol=1e-6)
    # dispatch is a 0/1 mask with exactly one slot per (kept) token.
    assert torch.all((dispatch == 0) | (dispatch == 1))
    assert int(dispatch.sum().item()) == t


def test_capacity_drop_is_accounted() -> None:
    """Below the busiest expert's load, the overflow is dropped and reported."""
    torch.manual_seed(2)
    d, e, t = 8, 2, 64
    # Force a heavy imbalance: a router that sends ~everything to expert 0.
    router = nn.Linear(d, e)
    with torch.no_grad():
        router.weight.zero_()
        router.bias.copy_(torch.tensor([10.0, 0.0]))
    experts = nn.ModuleList([Expert(d, 2) for _ in range(e)])
    x = torch.randn(t, d)
    capacity = 8  # << tokens routed to expert 0
    _, stats = moe_dispatch_combine(x, router, experts, capacity)
    assert stats["drop_frac"] > 0.0
    # expert 0 gets ~all tokens; only `capacity` survive -> drop ~ (load0 - cap)/T.
    _, _, route_stats = top1_route(router(x), e, capacity)
    load0 = float(route_stats["load"][0].item())
    assert math.isclose(stats["drop_frac"], max(load0 - capacity, 0) / t, rel_tol=1e-6)


def test_degenerate_router_has_low_active_fraction() -> None:
    """A collapsed router uses few experts -> active_experts well below E."""
    torch.manual_seed(3)
    d, e, t = 8, 8, 128
    router = nn.Linear(d, e)
    with torch.no_grad():
        router.weight.zero_()
        bias = torch.full((e,), -10.0)
        bias[0] = 10.0  # everything routes to expert 0
        router.bias.copy_(bias)
    _, _, stats = top1_route(router(torch.randn(t, d)), e, capacity=t)
    assert stats["active_experts"] == 1
    assert stats["active_experts"] / e < 0.70  # below the probe's min-active-frac gate


def test_balanced_router_uses_all_experts() -> None:
    """A well-mixed router activates every expert (the non-degenerate case)."""
    torch.manual_seed(4)
    d, e, t = 32, 4, 256
    router, _ = _build(d, e, ffn_mult=2, seed=4)
    _, _, stats = top1_route(router(torch.randn(t, d)), e, capacity=t)
    assert stats["active_experts"] == e


# --- multi-rank: expert-parallel must equal the local routed forward ---------- #
def _ep_two_rank_worker(rank: int, world: int, port: int, results) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world)
    # the probe's torchrun path keys on these (see _is_distributed)
    import os

    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world)
    try:
        d, e, t = 32, 4, 64
        router, experts = _build(d, e, ffn_mult=2, seed=7)  # identical on all ranks
        torch.manual_seed(100 + rank)                       # per-rank tokens
        x = torch.randn(t, d)
        out_ep, stats = moe_dispatch_combine(x, router, experts, capacity=t,
                                             expert_parallel=True)
        out_local, _ = moe_dispatch_combine(x, router, experts, capacity=t,
                                            expert_parallel=False)
        results[rank] = {
            "max_dev": float((out_ep - out_local).abs().max().item()),
            "drop_frac": stats["drop_frac"],
        }
    finally:
        dist.destroy_process_group()


def test_expert_parallel_two_ranks_matches_local_forward() -> None:
    """With identical experts on every rank and non-binding capacity, the
    all-to-all expert-parallel forward must reproduce each rank's local routed
    forward exactly: every source rank's slice must be processed by the owning
    expert (a path the world=1 identity all-to-all cannot exercise)."""
    import torch.multiprocessing as mp

    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_ep_two_rank_worker, args=(2, 29561, results), nprocs=2, join=True)
        res = dict(results)
    assert set(res) == {0, 1}
    for r in res.values():
        assert r["drop_frac"] == 0.0
        assert r["max_dev"] < 1e-5, r
