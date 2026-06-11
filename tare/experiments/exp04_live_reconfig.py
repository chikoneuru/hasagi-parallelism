"""exp04 — live mid-job layout reconfiguration: DDP -> reshard -> FSDP (scaffold).

Trains a model for one phase, performs a real state-transporting reshard with
``tare.state.reshard.ReshardController`` (data-parallel full replica -> fully
sharded), then continues training for a second phase, recording per-iteration
loss, the reconfiguration latency, and (on GPU) the reconfiguration energy. A
no-reshard control run with the same seed and data stream gives the trajectory
the reshard run must track: a correct reshard is loss-transparent, so the gate is

    max_i | loss_reshard[i] - loss_control[i] | <= --tol-ce  (default 0.3)

across the matched steps. The reshard is verify-before-commit: if the reassembled
state does not match the pre-reshard state the controller aborts and the model is
left untouched, which the harness reports as a failed reconfiguration.

Two modes. The default runs the transport single-process (CPU or one GPU),
which is exactly what the CPU test exercises (see
``tests/test_reshard_controller.py``). ``--live-rewrap`` (CUDA) runs the real
thing: phase 1 trains under an actual DDP wrapper, the measured window tears it
down and stands FSDP up over the verified state via
``tare.state.reshard.live_rewrap`` (optimizer state carried across), and
phase 2 continues under FSDP — with the reconfiguration latency and NVML
energy/power metered around the flip, plus optional extra FSDP<->DDP flip
cycles for latency/energy statistics. At world=1 FSDP degrades FULL_SHARD to
NO_SHARD (same wrapper code path, nothing to shard across) and the figures
contain no interconnect cost. Under torchrun the same flags shard across the
group for real, but note the cross-rank state movement in the measured flip is
the wrapper's own full-replica pipe (FSDP FULL_STATE_DICT gather + re-shard on
wrap, DDP rank-0 broadcast) — an upper bound on a flat-shard transport, not
one; certificate decisions are all-reduced so every rank commits or aborts
together.

Real run (2-GPU node)::

    torchrun --nproc_per_node=2 -m experiments.exp04_live_reconfig \\
        --live-rewrap --d-model 1024 --layers 8 --phase-iters 50 \\
        --out artifacts/exp04_live_reconfig.json

Single-GPU live run::

    python -m experiments.exp04_live_reconfig --live-rewrap \\
        --d-model 256 --layers 4 --phase-iters 50 --rewrap-cycles 5

Logic smoke-test (CPU)::

    python -m experiments.exp04_live_reconfig --smoke \\
        --d-model 64 --layers 2 --phase-iters 20 --to-world 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from tare.state.reshard import ReshardController, live_rewrap, wrap_layout

try:
    import pynvml
    _NVML = True
except Exception:  # pragma: no cover
    _NVML = False


def _build(d_model: int, layers: int, seed: int, device: torch.device):
    torch.manual_seed(seed)
    model = nn.Sequential(*[nn.Linear(d_model, d_model) for _ in range(layers)],
                          nn.Linear(d_model, 1)).to(device)
    return model


def _fixed_batches(n: int, batch: int, d_model: int, seed: int, device: torch.device):
    """A fixed sequence of (x, y) batches so reshard and control see identical data."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(batch, d_model, generator=g).to(device),
             torch.randn(batch, 1, generator=g).to(device)) for _ in range(n)]


def _train_segment(model, opt, batches, losses: list[float]) -> None:
    loss_fn = nn.MSELoss()
    for x, y in batches:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))


def _nvml_energy_j(local: int):
    if not _NVML or not torch.cuda.is_available():
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(local)
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
    except Exception:
        return None


def _nvml_handle(local: int):
    if not _NVML or not torch.cuda.is_available():
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(local)
    except Exception:
        return None


def _energy_j(handle) -> float | None:
    if handle is None:
        return None
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0  # mJ -> J
    except Exception:
        return None


def _power_w(handle) -> float | None:
    if handle is None:
        return None
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
    except Exception:
        return None


def _setup_pg() -> tuple[int, int, int, torch.device, bool]:
    """Join the torchrun process group, or make a 1-rank NCCL group standalone.

    Returns (rank, world, local, device, created); ``created`` says whether this
    call initialized the group (and so owns its teardown).
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        created = False
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl")
        created = True
    else:
        rank, world, local = 0, 1, 0
        dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29557",
                                rank=0, world_size=1)
        created = True
    torch.cuda.set_device(local)
    return rank, world, local, torch.device(f"cuda:{local}"), created


def _fresh_module(d_model: int, layers: int) -> nn.Module:
    """A skeleton for ``load_state_dict`` (values are overwritten); built under a
    forked RNG so mid-run construction never perturbs the training stream.
    No seeding inside the fork: ``manual_seed`` would also reset every CUDA
    generator, which ``fork_rng(devices=[])`` does not restore."""
    with torch.random.fork_rng(devices=[]):
        return nn.Sequential(*[nn.Linear(d_model, d_model) for _ in range(layers)],
                             nn.Linear(d_model, 1))


def _measured_flip(model, opt, layout_from: str, layout_to: str, world: int,
                   device: torch.device, handle, args, make_opt, flips: list[dict]):
    """One live rewrap with the reconfiguration window bracketed by
    synchronize + NVML reads. The energy counter advances at the driver's
    power-sampling cadence (tens of ms on GeForce), so a single ms-scale window
    can legitimately read 0 — power before/after and the cycle aggregate in the
    report are the corroborating signals."""
    torch.cuda.synchronize()
    p0 = _power_w(handle)
    e0 = _energy_j(handle)
    t0 = time.perf_counter()
    model, opt, cert = live_rewrap(
        model, opt, layout_from=layout_from, layout_to=layout_to,
        to_world=world, device=device, optim_factory=make_opt,
        module_factory=lambda: _fresh_module(args.d_model, args.layers),
        atol=args.atol, from_world=world,
    )
    torch.cuda.synchronize()
    latency_s = time.perf_counter() - t0
    e1 = _energy_j(handle)
    p1 = _power_w(handle)
    energy = (e1 - e0) if (e0 is not None and e1 is not None) else None
    flips.append({
        "direction": f"{layout_from}->{layout_to}",
        "latency_s": latency_s,
        "energy_counter_j": energy,
        # the counter ticks at the driver's power-sampling cadence; a whole-GPU
        # draw of >50 W cannot truly be 0 J over any window, so 0.0 == sub-tick
        "energy_counter_below_resolution": energy == 0.0,
        "power_w_sample_before": p0, "power_w_sample_after": p1,
        "stage_timings_s": cert.timings,
        "certificate": {
            "ok": cert.ok, "max_abs_diff": cert.max_abs_diff, "n_params": cert.n_params,
            "from_world": cert.from_world, "to_world": cert.to_world, "note": cert.note,
        },
    })
    return model, opt, cert


def _run_live(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("--live-rewrap requires CUDA (the FSDP wrapper has no CPU training path)")
        return 2
    import torch.distributed as dist

    rank, world, local, device, created = _setup_pg()
    handle = _nvml_handle(local)
    n = args.phase_iters
    seed = args.seed
    momentum = getattr(args, "momentum", 0.0)

    def make_opt(params):
        return torch.optim.SGD(params, lr=args.lr, momentum=momentum)

    try:
        # ---- reshard arm: DDP phase 1 -> measured live rewrap -> FSDP phase 2
        model = wrap_layout(_build(args.d_model, args.layers, seed, device), "ddp", device)
        opt = make_opt(model.parameters())
        batches = _fixed_batches(2 * n, args.batch, args.d_model, seed + 1, device)
        losses_reshard: list[float] = []
        _train_segment(model, opt, batches[:n], losses_reshard)

        flips: list[dict] = []
        model, opt, cert = _measured_flip(model, opt, "ddp", "fsdp", world, device,
                                          handle, args, make_opt, flips)
        _train_segment(model, opt, batches[n:], losses_reshard)

        # ---- control arm: same seed + data, plain module, never wrapped ------
        model_c = _build(args.d_model, args.layers, seed, device)
        opt_c = make_opt(model_c.parameters())
        losses_control: list[float] = []
        _train_segment(model_c, opt_c, batches, losses_control)

        max_dev = max(abs(a - b) for a, b in zip(losses_reshard, losses_control, strict=True))
        continuity_ok = max_dev <= args.tol_ce

        # ---- extra flip cycles: FSDP<->DDP statistics after the loss phases --
        # one long bracket around the back-to-back cycles gives an energy
        # window the counter can actually resolve when single flips are sub-tick
        torch.cuda.synchronize()
        cyc_e0, cyc_t0 = _energy_j(handle), time.perf_counter()
        n_cycle_flips = 0
        layout_from, layout_to = "fsdp", "ddp"
        for _ in range(int(getattr(args, "rewrap_cycles", 0))):
            model, opt, c = _measured_flip(model, opt, layout_from, layout_to, world,
                                           device, handle, args, make_opt, flips)
            n_cycle_flips += 1
            if not c.ok:    # globally agreed (all-reduced), so every rank breaks together
                break
            layout_from, layout_to = layout_to, layout_from
        torch.cuda.synchronize()
        cyc_e1, cyc_t1 = _energy_j(handle), time.perf_counter()

        # ---- cross-rank energy: the counter is per-GPU; a job-level J/flip at
        # world>1 must sum every rank's draw (collective: all ranks participate)
        if world > 1:
            for f in flips:
                per_rank: list = [None] * world
                dist.all_gather_object(per_rank, f["energy_counter_j"])
                f["energy_counter_j_per_rank"] = per_rank
                known = [e for e in per_rank if e is not None]
                f["energy_counter_j"] = sum(known) if known else None

        all_ok = all(f["certificate"]["ok"] for f in flips)
        lat = [f["latency_s"] for f in flips]
        known_e = [f["energy_counter_j"] for f in flips if f["energy_counter_j"] is not None]
        cycles_window = None
        if n_cycle_flips:
            cyc_e = (cyc_e1 - cyc_e0) if (cyc_e0 is not None and cyc_e1 is not None) else None
            if world > 1 and cyc_e is not None:
                per_rank = [None] * world
                dist.all_gather_object(per_rank, cyc_e)
                cyc_e = sum(e for e in per_rank if e is not None)
            cycles_window = {
                "window_s": cyc_t1 - cyc_t0, "flips_in_window": n_cycle_flips,
                "energy_counter_j": cyc_e,
                "energy_j_per_flip": (cyc_e / n_cycle_flips) if cyc_e is not None else None,
            }
        report = {
            "exp": "exp04-live-reconfig", "mode": "live-rewrap",
            "device": device.type, "world": world, "momentum": momentum,
            "d_model": args.d_model, "layers": args.layers, "phase_iters": n,
            # what the numbers mean (the artifact must not be mistaken for a
            # flat-shard-transport or interconnect measurement at world=1)
            "latency_includes": "snapshot + rank-local flat-plan transport+verify + "
                                "commit/wrap + post-rewrap gather+verify + optimizer rebuild",
            "transport_mechanism": "rank-local flat plan; cross-rank movement is the "
                                   "wrapper's own (FSDP full-state gather/shard, DDP rank0 "
                                   "broadcast) — an upper-bound pipe, not flat-shard transport; "
                                   "world=1 contains no interconnect cost",
            "energy_scope": "sum over ranks' whole-GPU NVML counters (per-GPU at world=1); "
                            "includes background draw — see power samples",
            "reshard_certificate": flips[0]["certificate"],
            "reconfig_latency_s": flips[0]["latency_s"],
            "reconfig_energy_j": flips[0]["energy_counter_j"],
            "loss_continuity": {
                "max_abs_deviation_vs_control": max_dev, "tol_ce": args.tol_ce,
                "ok": continuity_ok,
            },
            "loss_at_reshard_boundary": {
                "last_phase1": losses_reshard[n - 1], "first_phase2": losses_reshard[n],
            },
            "flips": flips,
            "flip_aggregate": {
                "count": len(flips), "all_certificates_ok": all_ok,
                "latency_s_mean": sum(lat) / len(lat),
                "latency_s_min": min(lat), "latency_s_max": max(lat),
                "energy_counter_j_total": sum(known_e) if known_e else None,
                "cycles_window": cycles_window,
            },
            "smoke": bool(args.smoke),
        }

        if rank == 0:
            print("=" * 64)
            print(f"exp04 live rewrap (world={world}, device={device}, "
                  f"layers={args.layers}, phase_iters={n}, momentum={momentum})")
            c0 = flips[0]["certificate"]
            print(f"  ddp->fsdp certificate: ok={c0['ok']} "
                  f"max_abs_diff={c0['max_abs_diff']:.2e} ({c0['n_params']} params) {c0['note']}")
            print(f"  reconfig latency: {flips[0]['latency_s'] * 1000:.2f} ms; "
                  f"counter energy: {flips[0]['energy_counter_j']} J")
            print(f"  loss continuity vs control: max dev {max_dev:.3e} "
                  f"(tol {args.tol_ce}) -> {'OK' if continuity_ok else 'FAIL'}")
            if len(flips) > 1:
                agg = report["flip_aggregate"]
                print(f"  {agg['count']} flips: latency mean {agg['latency_s_mean'] * 1000:.2f} ms "
                      f"[{agg['latency_s_min'] * 1000:.2f}, {agg['latency_s_max'] * 1000:.2f}]; "
                      f"counter energy total {agg['energy_counter_j_total']} J; "
                      f"all certs ok: {agg['all_certificates_ok']}")
            if args.out:
                from pathlib import Path
                p = Path(args.out)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(report, indent=2))
                print(f"wrote {p}")
        return 0 if (all_ok and continuity_ok) else 1
    finally:
        if created:
            dist.destroy_process_group()


def run(args: argparse.Namespace) -> int:
    if getattr(args, "live_rewrap", False):
        return _run_live(args)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    n = args.phase_iters
    seed = args.seed

    # ---- reshard run: phase 1 -> reshard -> phase 2 -----------------------
    model = _build(args.d_model, args.layers, seed, device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    batches = _fixed_batches(2 * n, args.batch, args.d_model, seed + 1, device)
    losses_reshard: list[float] = []
    _train_segment(model, opt, batches[:n], losses_reshard)

    rc = ReshardController(atol=args.atol)
    rc.capture(model, from_world=1)
    e0 = _nvml_energy_j(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    cert = rc.reshard_and_commit(model, to_world=args.to_world)
    if device.type == "cuda":
        torch.cuda.synchronize()
    reconfig_latency_s = time.perf_counter() - t0
    e1 = _nvml_energy_j(0)
    reconfig_energy_j = (e1 - e0) if (e0 is not None and e1 is not None) else None

    _train_segment(model, opt, batches[n:], losses_reshard)

    # ---- control run: same seed + data, no reshard ------------------------
    model_c = _build(args.d_model, args.layers, seed, device)
    opt_c = torch.optim.SGD(model_c.parameters(), lr=args.lr)
    losses_control: list[float] = []
    _train_segment(model_c, opt_c, batches, losses_control)

    max_dev = max(abs(a - b) for a, b in zip(losses_reshard, losses_control, strict=True))
    continuity_ok = max_dev <= args.tol_ce

    report = {
        "exp": "exp04-live-reconfig", "device": device.type,
        "d_model": args.d_model, "layers": args.layers, "phase_iters": n,
        "to_world": args.to_world,
        "reshard_certificate": {
            "ok": cert.ok, "max_abs_diff": cert.max_abs_diff, "n_params": cert.n_params,
            "from_world": cert.from_world, "to_world": cert.to_world, "note": cert.note,
        },
        "reconfig_latency_s": reconfig_latency_s,
        "reconfig_energy_j": reconfig_energy_j,
        "loss_continuity": {
            "max_abs_deviation_vs_control": max_dev, "tol_ce": args.tol_ce, "ok": continuity_ok,
        },
        "loss_at_reshard_boundary": {
            "last_phase1": losses_reshard[n - 1], "first_phase2": losses_reshard[n],
        },
        "smoke": bool(args.smoke),
    }

    print("=" * 64)
    print(f"exp04 live reconfig (device={device.type}, layers={args.layers}, "
          f"phase_iters={n}, to_world={args.to_world})")
    if args.smoke:
        print("  [SMOKE TEST — single-process logic; real DDP/FSDP + energy need torchrun + GPUs]")
    print(f"  reshard certificate: ok={cert.ok} max_abs_diff={cert.max_abs_diff:.2e} "
          f"({cert.n_params} params) {cert.note}")
    print(f"  reconfig latency: {reconfig_latency_s * 1000:.2f} ms; "
          f"energy: {reconfig_energy_j if reconfig_energy_j is not None else 'n/a (CPU)'}")
    print(f"  loss continuity vs control: max dev {max_dev:.3e} "
          f"(tol {args.tol_ce}) -> {'OK' if continuity_ok else 'FAIL'}")
    if args.out:
        from pathlib import Path
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
        print(f"wrote {p}")
    return 0 if (cert.ok and continuity_ok) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--phase-iters", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--to-world", type=int, default=2, help="target shard count for the FSDP layout")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-6, help="reshard verification tolerance")
    p.add_argument("--tol-ce", type=float, default=0.3, help="max allowed loss deviation vs control")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--live-rewrap", action="store_true",
                   help="real DDP->FSDP wrapper flip on CUDA (vs the in-memory transport)")
    p.add_argument("--rewrap-cycles", type=int, default=3,
                   help="extra measured FSDP<->DDP flips for latency/energy statistics")
    p.add_argument("--momentum", type=float, default=0.0,
                   help="SGD momentum (exercises optimizer-state carry across the rewrap)")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
