# HASAGI Testbed

Reference implementation for **HASAGI — Hybrid Architecture for Serverless AI Training under Grid Intensity**.

> **What is actually measured vs modelled (read first).** On the single-GPU
> testbed (RTX 3080 Ti) the **energy** results are real NVML measurements; **carbon**
> is a derived proxy (energy × grid intensity). The hybrid-parallel partitioner and
> the ZeRO-style state redistribution are **algorithm + simulation only — there is no
> real `torch.distributed`/pipeline execution** in this version, and multi-GPU
> training is future work. Workloads in the power-cap study are small synthetic
> microbenchmarks (a TinyResNet and a small Transformer), not full training runs.
> Claims are scoped accordingly throughout; see "Measured findings" below.

## Layout

```
hasagi/                  Python package (the framework)
├── orchestrator/      Job orchestrator + control loop (FastAPI)
├── parallel/          Hybrid parallel controller — HyPAS Algo 1+2, Hydrozoa-style planner
├── admission/         ElasticFlow MSS + Energy-Adjusted MSS
├── energy/            Carbon trace replay, ElectricityMaps client, scheduling policies
├── pool/              GPU burst pool manager (local Docker + Knative stub)
├── state/             Fault-tolerant state (Redis + checkpoint)
├── worker/            Training worker (PyTorch + elastic + pipeline)
├── metrics/           Prometheus exporters
├── models/            Benchmark model zoo (ResNet, ViT, GPT-2)
└── data/              Dataset loaders
traces/                Carbon intensity traces (synthetic + ElectricityMaps replay)
experiments/           Reproducible experiment scripts
tests/                 Pytest unit tests (algorithms; no GPU needed)
docker/                Dockerfiles
k8s/                   Kubernetes/Knative manifests (cluster mode)
```

## Three ways to run

### 1. Smoke test (no GPU, no Docker)

The core algorithms are pure Python and can be exercised by unit tests + a simulation experiment:

```bash
cd src
make venv                  # creates .venv + installs dev deps (CPU torch)
make test                  # unit tests across partitioner, MSS, carbon policy, control loop
make lint                  # ruff check
make smoke                 # exp01: 1-job control-loop simulation
```

Or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e .[dev]
pytest -ra
```

Expected: unit tests pass and the smoke test prints a 24h schedule trace with allocation changes at carbon-peak hours.

### 2. Local stack (Docker)

Bring up orchestrator + Redis + Prometheus + worker stubs on one machine:

```bash
make up                  # docker-compose up -d
make exp02               # carbon-replay experiment (ResNet-18 / CIFAR-10)
make logs                # tail orchestrator
make down
```

### 3. Cluster mode (Kubernetes)

Manifests under `k8s/` are scaffolds — flesh out for your cluster:

```bash
kubectl apply -f k8s/
hasagi-cli submit --model resnet18 --dataset cifar10 \
                --deadline 4h --carbon-budget 0.5kg
```

## Key dependencies

| Component | Library |
|---|---|
| Training framework | PyTorch ≥ 2.1, torchvision |
| Elastic runtime | `torch.distributed.elastic` |
| Orchestrator API | FastAPI + uvicorn |
| Param store | Redis |
| Metrics | prometheus-client |
| Carbon data | ElectricityMaps API (live) or trace replay (offline) |
| RL policy (optional) | Stable-Baselines3 + Gymnasium |
| Container runtime | Docker (local) / Knative + K8s (cluster) |

## Development

```bash
make lint               # ruff check hasagi tests experiments
make test               # pytest -ra
.venv/bin/pytest -x     # fast-fail mode
.venv/bin/mypy hasagi/    # type check (informational)
```

CI runs the same lint + test + smoke-experiment sequence on every push to `main` and on every PR (see [`.github/workflows/test.yml`](.github/workflows/test.yml)).

## Status

What's implemented vs stubbed:

| Module | State |
|---|---|
| `parallel/partitioner.py` — PipeDream k-way pipeline partitioner (O(n²·K) DP + incremental sliding-window) | ✅ bottleneck-min + energy-per-iter objectives, k-1 cut incremental variant, memory + power-cap feasibility constraints, stagnation tracker |
| `parallel/inter_batch.py` — 1F1B + deficit-WRR scheduler (Katevenis-Sidiropoulos JSAC'91 + PipeDream) | ✅ FLOPS-weighted baseline + R1/R2/R3 rules + stage_id-keyed; energy-aware WRR + power-slack guard with live telemetry refresh |
| `parallel/planner.py` (Hydrozoa hybrid strategy) | ✅ implemented |
| `admission/mss.py` (ElasticFlow MSS + **EnergyBudgetMSS** + marginal-energy allocator) | ✅ implemented (energy as primary budget, carbon proxy optional) |
| `energy/telemetry.py` (NVML + RAPL + aggregator + Prometheus pusher) | ✅ background-thread polling, dependency-injectable for CI |
| `energy/carbon_sources.py` (ElectricityMaps + WattTime + IEA static + multi-source aggregator) | ✅ implemented |
| `energy/carbon_trace.py` (proxy-only trace replay) | ✅ implemented |
| `energy/policy.py` (rule-based + PowerAwareRule + MPC with reconfig penalty) | ✅ implemented |
| `energy/rl_policy.py` (PPO) | 🚧 scaffold; training pending |
| `orchestrator/control_loop.py`, `energy_aware_control_loop.py`, `api.py` | ✅ implemented |
| `pool/local_pool.py` (Docker) | ✅ |
| `pool/knative_pool.py` | 🚧 stub |
| `worker/trainer.py` | ✅ single-node training; elastic + power-cap planned |
| `state/redis_store.py`, `state/checkpoint.py` | ✅ minimal |

**Energy vs carbon in this codebase**: kWh is the primary metric throughout the API
(`EnergyBudgetMSS`, the `energy/` package name, the experiment outputs). Carbon enters
only as an *optional* proxy budget on top of the energy budget, computed by multiplying
projected energy by a grid intensity trace. Energy is measured directly (NVML/RAPL,
~±2% noise) while carbon is a proxy with explicit uncertainty bounds — the codebase
flags >20% disagreement between carbon sources rather than cherry-picking a single
intensity value.

Tenplex-style PTC state redistribution and end-to-end training-loop integration with
real NVML on a multi-GPU testbed are explicit follow-ups.

## Measured findings (single-GPU, honest scope)

These are the defensible results on the current testbed. Energy is measured (NVML);
carbon is energy × a grid-intensity trace. Numbers carry their caveats.

- **Carbon-aware throttle vs pause, on 16 real ElectricityMaps zones × 2 seasons**
  ([`exp_realtrace_pareto.py`](experiments/exp_realtrace_pareto.py),
  [`exp_realtrace_sensitivity.py`](experiments/exp_realtrace_sensitivity.py)). Against a
  GREEN-style temporal-shifter ported across its full capability range, throttling
  to the energy-optimal cap and pausing are **carbon-comparable at a matched budget**
  (zone-clustered fair gap +0.57 pp, not significant); the carbon winner is
  zone-dependent. The robust advantage of throttle over pause for *training* is
  **latency** — a 24 h job finishes ~+2 h late under throttle vs ~+14 h under
  deferral. **For training, throttle, don't pause.**
- **Decomposition: most of the saving is not carbon-awareness.** Throttle's ~+9 %
  carbon vs always-full splits into ~+7.4 % same-budget *carbon-blind* cap efficiency
  (established prior art: Zeus, Perseus/EnvPipe) plus only **+1.56 pp [+0.99, +2.15]**
  attributable to the carbon *signal* (knowing which windows are dirty), Holm-robust
  at the headline operating point. The carbon signal is the genuine, modest contribution.
- **Energy-optimal power cap (repeated DVFS sweep, error bars + both cap-orders)**
  ([`exp_hardware_pareto.py`](experiments/exp_hardware_pareto.py) `--repeats --cap-order`,
  [`exp_cap_robustness.py`](experiments/exp_cap_robustness.py)). The energy-per-iter
  U-curve has a **broad, order-invariant flat bottom** (ResNet ~200–250 W, the
  Transformer ~250–300 W) whose plateaus **overlap at ~250 W** — so an earlier
  single-sweep "workload-dependent optimum (200 vs 250 W)" claim **did not survive
  repeats and is withdrawn**. The surviving effect is **under-capping risk**: a
  too-low cap penalises the compute-heavy Transformer (~+17 % at 200 W, ~+73 % at
  150 W) while ResNet tolerates it. A cap in the ~250–300 W overlap serves both.
- **Co-tenant contention: measured throughput floor + partitioner decision-quality**
  ([`exp_cotenant_contention.py`](experiments/exp_cotenant_contention.py),
  [`exp_contention_decision.py`](experiments/exp_contention_decision.py),
  [`exp_comm_sensitivity.py`](experiments/exp_comm_sensitivity.py)). *Measured:*
  time-sliced co-location of N identical GPU-bound ResNet trainers on the 3080 Ti
  (own processes, 3 repeats) — each tenant retains **c(2)=0.46, c(3)=0.31** of solo
  throughput, just below the 1/N time-slice floor, so aggregate GPU throughput stays
  flat at ~0.92× (an ≈8 % co-location overhead beyond pure time-slicing). This is
  time-sliced sharing; an MPS daemon would reduce contention, so the measured c is a
  lower bound. *Simulation (algorithm only):* the partitioner's decisions are robust
  to **uniform** slowdown (scale-invariant, ~0 regret) but a **contention-blind** plan
  under **asymmetric** co-tenancy leaves regret growing to ~100 % at c=0.3, which the
  implemented sliding-window incremental re-partition recovers (1–5 steps, with a
  StagnationTracker → full-DP fallback for window-trapping cases); the chosen
  parallel strategy and pipeline cuts are also bandwidth-regime dependent (a
  bandwidth-blind decision costs up to +60–128 % in the analytic cost model). Regret
  here is modelled, not wall-clock; there is no distributed execution.
- **Reproducibility note.** Result artifacts and downloaded traces live under
  gitignored `artifacts/` and `data_cache/`; regenerate the carbon traces with
  `experiments/fetch_electricitymaps_traces.py` (needs `$ELECTRICITYMAPS_TOKEN`) and
  the GPU sweeps with `exp_hardware_pareto.py` (needs `nvidia-smi -pl` privileges).

**Not yet established (future work):** real multi-GPU / distributed execution; the
hybrid-parallel re-partition and ZeRO redistribution beyond algorithm + simulation;
any carbon claim on real training of large models.
