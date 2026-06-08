# hasagi-parallelism

A research monorepo on **energy- and carbon-aware, elastic serverless DNN training**. Three
independent projects share this repository:

| Directory | Codename | Focus |
|---|---|---|
| [`tare/`](tare/) | **Tare** | A measurement-grounded study of carbon-aware serverless training: a falsification map across three levers (throttle the power cap, pause/defer, repartition the hybrid-parallel layout), plus a reusable break-even instrument. Mature codebase. |
| [`hasagi/`](hasagi/) | **ATTEST** | Verified elastic reconfiguration: a machine-checkable *equivalence certificate* for the state-transport transition during a parallelism reshard (verify-before-commit, abort-to-last-verified). Early-stage scaffold. |
| [`green/`](green/) | **IdleLedger** | Idle-/provisioning-energy-aware serverless training: consolidate-then-power-down freed accelerators, gated by a checkpoint-reload break-even model. Early-stage scaffold. |

`tare/` is the mature codebase (package, experiments, tests, result artifacts). `hasagi/` and
`green/` currently hold kill-test scaffolds for their respective research directions.

## Layout

```
.
├── tare/      # Tare: the `tare` Python package + experiments/ tests/ artifacts/ docker/ k8s/
├── hasagi/    # ATTEST: attest_kill_test.py (equivalence-certificate + fault-suite)
├── green/     # IdleLedger: idle_breakeven_bench.py (+ artifacts/)
└── .github/   # CI (runs lint + tests + smoke experiments inside tare/)
```

## `tare/` — build & test

```bash
cd tare
pip install -e .[dev]                 # installs the `tare` package (CPU torch is fine for tests)
pytest -ra                            # unit tests
ruff check tare tests experiments     # lint
python -m experiments.exp01_smoke_test   # run any experiment as a module (from tare/)
```

Result artifacts live in `tare/artifacts/` (the JSON each reported number/figure reads).

## `hasagi/` — ATTEST kill-test

```bash
python hasagi/attest_kill_test.py     # CPU-only; certifies a clean reshard and catches a fault suite
```

A clean reshard certifies cleanly; an injected reconfiguration fault (dropped shard, stale optimizer
slot, lost microbatch, wrong reduction order) is caught and rolled back to the last verified state.

## `green/` — IdleLedger break-even bench

```bash
python green/idle_breakeven_bench.py --gpu 0 --sizes 125M,350M   # needs a CUDA GPU + nvidia-ml-py
```

Measures active vs idle GPU power and checkpoint-reload energy, then computes the break-even idle
duration `t* = E_reload / (P_idle - P_powered_down)` against a target idle gap.

## Naming

- The `tare/` package import name is **`tare`** (renamed from an earlier `hasagi`).
- The bare name **`HASAGI`** is reserved for the `hasagi/` project.
- The tag **`pre-restructure-tare`** marks the original single-project layout, before this monorepo.

## Conventions

- The default branch is `main`.
- CI runs inside `tare/` (the test workflow's working directory).
- Internal research notes and plans are kept local and are not tracked in this repository.
