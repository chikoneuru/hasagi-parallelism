"""Multi-seed PPO evaluation for the keep/drop decision.

Trains PPO from scratch on each of ``--seeds`` seeds, evaluates each model
against static-max + rule-based + MPC on a held-out trace, and reports
mean ± stddev energy/iters per policy plus Cohen's d for PPO vs MPC.

Decision rule:
    KEEP PPO iff
        - PPO mean energy < MPC mean energy by ≥ 5% (one-sided),
        - Cohen's d for the energy delta ≥ 1.5 (large effect).
    Otherwise DROP and reframe co-design as MPC-only.

Single-trial mode (``--seeds 1``) is fine for smoke / convergence check; the
full keep/drop decision requires the default ``--seeds 5``.

Usage:
    python experiments/exp04b_ppo_eval_q2.py --seeds 5 --timesteps 200000
    python experiments/exp04b_ppo_eval_q2.py --seeds 5 --timesteps 50000  # faster
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hasagi.energy.carbon_trace import CarbonTrace, load_csv_trace, synthetic_solar_trace
from hasagi.energy.policy import MPCPolicy, RuleBasedPolicy
from hasagi.energy.rl_env import EnvConfig, HasagiCarbonEnv


@dataclass
class TrialResult:
    seed: int
    policy: str
    energy_kwh: float
    iters_done: int
    emissions_g: float


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Pooled-standard-deviation Cohen's d for two independent samples.

    Returns ``inf`` if both groups have zero variance (degenerate; ranks the
    sign of the mean difference).
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    mean_a, mean_b = statistics.mean(group_a), statistics.mean(group_b)
    var_a, var_b = statistics.variance(group_a), statistics.variance(group_b)
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return float("inf") if mean_a != mean_b else 0.0
    return (mean_a - mean_b) / pooled


def train_one_seed(
    seed: int,
    cfg: EnvConfig,
    trace: CarbonTrace,
    timesteps: int,
    n_envs: int,
    n_steps: int,
):
    """Train and return a PPO model on a vec-env seeded by ``seed``."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv(
        [
            lambda i=i: Monitor(HasagiCarbonEnv(cfg, trace=trace, seed=seed * 100 + i))
            for i in range(n_envs)
        ]
    )
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model


def rollout(
    cfg: EnvConfig,
    trace: CarbonTrace,
    seed: int,
    name: str,
    decide_fn,
) -> TrialResult:
    env = HasagiCarbonEnv(cfg, trace=trace, seed=seed)
    obs, _ = env.reset()
    cur = cfg.min_gpus
    sim_t = 0.0
    energy = 0.0
    iters = 0
    emissions = 0.0
    while True:
        intensity = trace.intensity_at(sim_t)
        forecast = [
            (t, trace.intensity_at(sim_t + t))
            for t in range(0, cfg.tick_seconds * 6, cfg.tick_seconds)
        ]
        tgt = decide_fn(obs, cur, intensity, forecast)
        tgt = max(cfg.min_gpus, min(cfg.max_gpus, int(tgt)))
        obs, _, term, trunc, info = env.step(tgt - cfg.min_gpus)
        emissions += info["delta_kwh"] * intensity
        cur = info["current_gpus"]
        sim_t = info["sim_t"]
        energy = info["energy_used_kwh"]
        iters = info["iters_done"]
        if term or trunc:
            break
    return TrialResult(seed=seed, policy=name, energy_kwh=energy,
                       iters_done=iters, emissions_g=emissions)


def evaluate_seed(
    seed: int,
    cfg: EnvConfig,
    trace: CarbonTrace,
    timesteps: int,
    n_envs: int,
    n_steps: int,
    console: Console,
) -> list[TrialResult]:
    t0 = time.monotonic()
    console.print(f"[bold]Seed {seed}[/]: training PPO ({timesteps} steps × {n_envs} envs)...")
    model = train_one_seed(seed, cfg, trace, timesteps, n_envs, n_steps)
    train_s = time.monotonic() - t0

    # Reusable decision functions
    rule = RuleBasedPolicy(min_gpus=cfg.min_gpus, max_gpus=cfg.max_gpus)
    mpc = MPCPolicy(
        min_gpus=cfg.min_gpus, max_gpus=cfg.max_gpus,
        horizon_steps=6, step_seconds=cfg.tick_seconds,
        power_per_gpu_w=cfg.power_per_gpu_w,
        throughput_per_gpu=lambda g: cfg.base_throughput_iters_per_s * (g ** cfg.scaling_efficiency),
        iterations_remaining=cfg.target_iters,
        deadline_seconds_remaining=cfg.deadline_seconds,
    )
    results = []
    for name, fn in (
        ("static-max", lambda *_: cfg.max_gpus),
        ("rule-based", lambda obs, cur, intensity, _: rule.decide(cur, intensity).target_gpus),
        ("mpc", lambda obs, cur, intensity, forecast: mpc.decide(cur, forecast).target_gpus),
        ("ppo", lambda obs, *_: int(model.predict(obs, deterministic=True)[0]) + cfg.min_gpus),
    ):
        results.append(rollout(cfg, trace, seed, name, fn))

    eval_s = time.monotonic() - t0 - train_s
    console.print(
        f"[dim]  seed {seed}: train {train_s:.1f}s + eval {eval_s:.1f}s "
        f"= {time.monotonic() - t0:.1f}s[/]"
    )
    return results


def aggregate_and_decide(all_results: list[TrialResult], console: Console) -> dict:
    """Group by policy, compute mean / stddev / Cohen's d vs MPC, print + decide."""
    by_policy: dict[str, list[TrialResult]] = {}
    for r in all_results:
        by_policy.setdefault(r.policy, []).append(r)

    static_mean = statistics.mean(r.energy_kwh for r in by_policy["static-max"])
    mpc_energies = [r.energy_kwh for r in by_policy["mpc"]]
    ppo_energies = [r.energy_kwh for r in by_policy["ppo"]]
    mpc_mean = statistics.mean(mpc_energies)
    ppo_mean = statistics.mean(ppo_energies)

    table = Table(title="Q2 decide datum — PPO vs baselines (multi-seed)")
    table.add_column("policy")
    table.add_column("mean kWh", justify="right")
    table.add_column("stddev", justify="right")
    table.add_column("Δ vs static", justify="right")
    table.add_column("Δ vs mpc", justify="right")
    table.add_column("iters (mean)", justify="right")
    for name in ("static-max", "rule-based", "mpc", "ppo"):
        rs = by_policy[name]
        es = [r.energy_kwh for r in rs]
        its = [r.iters_done for r in rs]
        mean_e = statistics.mean(es)
        sd_e = statistics.stdev(es) if len(es) > 1 else 0.0
        delta_s = (mean_e / max(static_mean, 1e-9) - 1.0) * 100.0
        delta_m = (mean_e / max(mpc_mean, 1e-9) - 1.0) * 100.0
        table.add_row(
            name,
            f"{mean_e:.3f}",
            f"{sd_e:.3f}",
            f"{delta_s:+.1f}%",
            f"{delta_m:+.1f}%",
            f"{statistics.mean(its):,.0f}",
        )
    console.print(table)

    # MPC is the reference; PPO must improve to be kept.
    d = cohens_d(mpc_energies, ppo_energies)   # positive if MPC > PPO (PPO wins)
    delta_pct = (mpc_mean - ppo_mean) / max(mpc_mean, 1e-9) * 100.0

    decision = {
        "n_seeds": len(mpc_energies),
        "ppo_mean_kwh": ppo_mean,
        "mpc_mean_kwh": mpc_mean,
        "delta_pct_ppo_vs_mpc": delta_pct,
        "cohens_d_mpc_minus_ppo": d,
        "keep_ppo": delta_pct >= 5.0 and d >= 1.5,
    }
    console.print(
        f"\n[bold]Q2 decision[/]: "
        f"PPO energy = MPC × {ppo_mean/mpc_mean:.3f} (Δ {delta_pct:+.1f}%), "
        f"Cohen's d = {d:.2f}"
    )
    if decision["keep_ppo"]:
        console.print("[bold green]→ KEEP PPO[/] (≥5% energy reduction vs MPC AND |d| ≥ 1.5)")
    else:
        console.print(
            f"[bold yellow]→ DROP PPO[/] (insufficient improvement vs MPC: need ≥5% AND |d| ≥ 1.5, "
            f"got {delta_pct:+.1f}% / d={d:.2f}). Reframe co-design as MPC-only."
        )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--target-iters", type=int, default=200_000)
    parser.add_argument("--episode-hours", type=int, default=24)
    parser.add_argument("--energy-budget-kwh", type=float, default=5.0)
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--output", default="artifacts/q2_results.json")
    args = parser.parse_args()

    console = Console()
    trace = (
        load_csv_trace(args.trace) if args.trace
        else synthetic_solar_trace(hours=int(args.episode_hours) + 1)
    )
    cfg = EnvConfig(
        min_gpus=args.min_gpus,
        max_gpus=args.max_gpus,
        target_iters=args.target_iters,
        deadline_seconds=args.episode_hours * 3600.0,
        energy_budget_kwh=args.energy_budget_kwh,
    )

    console.print(
        f"[bold]Multi-seed PPO evaluation[/]: {args.seeds} seeds × {args.timesteps} steps "
        f"× {args.n_envs} envs"
    )
    all_results: list[TrialResult] = []
    t0 = time.monotonic()
    for s in range(args.seeds):
        all_results.extend(evaluate_seed(
            s, cfg, trace, args.timesteps, args.n_envs, args.n_steps, console,
        ))
    total_s = time.monotonic() - t0
    console.print(f"\n[bold green]All seeds complete[/]: {total_s:.1f} s")

    decision = aggregate_and_decide(all_results, console)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "results": [asdict(r) for r in all_results],
        "decision": decision,
    }, indent=2))
    console.print(f"\n[dim]Results saved: {out}[/]")


if __name__ == "__main__":
    main()
