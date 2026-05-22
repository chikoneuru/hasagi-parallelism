"""Train a PPO scaling policy against the carbon-aware HiseCarbonEnv.

Pipeline:
    1. Build a Gym env over the synthetic-solar trace (default) or a user-supplied
       CSV trace (e.g., ElectricityMaps export converted to ``intensity_g_per_kwh``).
    2. Train Stable-Baselines3 PPO for ``--timesteps`` env steps.
    3. Save the trained model to ``artifacts/ppo_hise.zip`` (gitignored).
    4. Optionally evaluate against the MPC + rule-based baselines (``--eval``)
       and print an energy-delta table — this is the Q2 keep-or-drop datum.

The ``--quick`` flag truncates training and shrinks the episode horizon so the
script returns in under a minute, suitable for CI smoke or convergence shape
inspection. Production training uses ``--timesteps 200_000`` (default) on a
24-hour episode.

Usage:
    python experiments/exp04_ppo_train.py --quick                # ~30s, 2 envs
    python experiments/exp04_ppo_train.py --timesteps 200000     # full run
    python experiments/exp04_ppo_train.py --trace traces/em_us_west.csv --eval
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from hise.energy.carbon_trace import CarbonTrace, load_csv_trace, synthetic_solar_trace
from hise.energy.rl_env import EnvConfig, HiseCarbonEnv


def make_env(cfg: EnvConfig, trace: CarbonTrace, seed: int = 0):
    def _thunk():
        env = HiseCarbonEnv(cfg, trace=trace, seed=seed)
        return env
    return _thunk


def train(args: argparse.Namespace) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

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

    n_envs = args.n_envs
    vec_env = DummyVecEnv(
        [lambda i=i: Monitor(HiseCarbonEnv(cfg, trace=trace, seed=i)) for i in range(n_envs)]
    )

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1 if args.verbose else 0,
        seed=args.seed,
    )

    t0 = time.monotonic()
    console.print(
        f"[bold]Training PPO[/]: {args.timesteps} timesteps × {n_envs} envs "
        f"on trace ({'CSV' if args.trace else 'synthetic-solar'})"
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    train_s = time.monotonic() - t0
    console.print(f"[bold green]Training complete[/]: {train_s:.1f} s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    console.print(f"[bold]Saved model[/]: {out_path}")
    return out_path


def evaluate(args: argparse.Namespace, model_path: Path) -> None:
    """Roll out the trained PPO against rule-based + MPC on a held-out trace.

    Reports energy + emissions + iters achieved per policy. Q2 keep/drop datum:
    PPO must reduce kWh by ≥ 5% vs MPC over n=5 trials with non-overlapping
    confidence intervals (Cohen's d ≥ 1.5 per the pre-registered power analysis).
    Single-trial smoke run only here — full analysis lives in eval_q2.py once
    --timesteps reaches the design value.
    """
    from stable_baselines3 import PPO

    from hise.energy.policy import MPCPolicy, RuleBasedPolicy

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

    rule = RuleBasedPolicy(min_gpus=args.min_gpus, max_gpus=args.max_gpus)
    mpc = MPCPolicy(
        min_gpus=args.min_gpus,
        max_gpus=args.max_gpus,
        horizon_steps=6,
        step_seconds=cfg.tick_seconds,
        power_per_gpu_w=cfg.power_per_gpu_w,
        throughput_per_gpu=lambda g: cfg.base_throughput_iters_per_s * (
            g ** cfg.scaling_efficiency
        ),
        iterations_remaining=cfg.target_iters,
        deadline_seconds_remaining=cfg.deadline_seconds,
    )
    model = PPO.load(str(model_path))

    results: list[tuple[str, float, int, float]] = []

    def _rollout(name: str, decide_fn) -> tuple[str, float, int, float]:
        env = HiseCarbonEnv(cfg, trace=trace, seed=args.seed)
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
        return name, energy, iters, emissions

    results.append(_rollout("static-max", lambda *_: args.max_gpus))
    results.append(_rollout("rule-based", lambda obs, cur, intensity, _: rule.decide(cur, intensity).target_gpus))
    results.append(_rollout("mpc", lambda obs, cur, intensity, forecast: mpc.decide(cur, forecast).target_gpus))
    results.append(_rollout(
        "ppo",
        lambda obs, *_: int(model.predict(obs, deterministic=True)[0]) + cfg.min_gpus,
    ))

    table = Table(title="Q2 decide datum — PPO vs rule-based vs MPC (single trial)")
    table.add_column("policy")
    table.add_column("energy (kWh)", justify="right")
    table.add_column("Δ vs static", justify="right")
    table.add_column("Δ vs mpc", justify="right")
    table.add_column("emissions (g)", justify="right")
    table.add_column("iters done", justify="right")
    static_kwh = results[0][1]
    mpc_kwh = next(r[1] for r in results if r[0] == "mpc")
    for name, kwh, iters, emis in results:
        delta_s = (kwh / max(static_kwh, 1e-9) - 1.0) * 100.0
        delta_m = (kwh / max(mpc_kwh, 1e-9) - 1.0) * 100.0
        table.add_row(
            name,
            f"{kwh:.3f}",
            f"{delta_s:+.1f}%",
            f"{delta_m:+.1f}%",
            f"{emis:.0f}",
            f"{iters:,}",
        )
    console = Console()
    console.print(table)
    console.print(
        "\n[dim]Single trial only — full Q2 statistical test (n=5, Cohen's d ≥ 1.5, "
        "Bonferroni α=0.00083) requires repeated runs over multiple seeds.[/]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=None, help="CSV trace; default = synthetic solar")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-iters", type=int, default=200_000)
    parser.add_argument("--episode-hours", type=int, default=24)
    parser.add_argument("--energy-budget-kwh", type=float, default=5.0)
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="artifacts/ppo_hise.zip")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--eval", action="store_true",
                        help="After training, evaluate vs rule-based + MPC")
    parser.add_argument("--quick", action="store_true",
                        help="Short training (10k steps, 2 envs, 6h episode) for smoke / CI")
    args = parser.parse_args()

    if args.quick:
        args.timesteps = min(args.timesteps, 10_000)
        args.n_envs = 2
        args.n_steps = 256
        args.episode_hours = 6
        args.target_iters = 50_000

    np.random.seed(args.seed)
    model_path = train(args)
    if args.eval:
        evaluate(args, model_path)


if __name__ == "__main__":
    main()
