# HASAGI Testbed

Reference implementation for **HASAGI — Hybrid Parallelism Architecture for Energy-Aware Serverless AI Training**.

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
