"""Price the cheapest competing oracle: a one-batch forward eval after the transition.

The obvious objection to certifying state transitions is that a trivial
baseline — run one deterministic batch through the restored model and compare
against a reference output captured at save time — might catch the same
corruption for free. This experiment measures that baseline honestly on both
reproduced natural bugs (DeepSpeed#6791 conversion corruption, Megatron-LM
PR#520 SwiGLU mis-split) and on five designed corruptions drawn from the
documented bug classes in ``docs/reshard-bug-corpus.md``, then runs the
transition certificate on the identical states.

The eval oracle is given its STRONGEST form: the same fixed batch, bit-exact
reference outputs, no data nondeterminism, and two thresholds on the relative
output deviation — strict 1e-3 (better than any real resume-loss check) and
realistic 1e-2 (run-to-run loss noise). What it structurally cannot see:

  * optimizer-state corruption (the Megatron-LM#761 merge/reshard class) and
    step-counter resets do not change the forward pass at all — output
    deviation is exactly 0.0 while the next optimizer steps are silently
    mis-scaled;
  * trainer-progress loss lives outside the model entirely;
  * near-tolerance value damage (narrow-dtype round-trips, small additive
    shard-boundary perturbations) can sit below any noise floor a loss check
    must tolerate.

The designed cases are clearly labeled as injected demonstrations of those
documented classes; they are not part of the natural-bug claim (the two
natural cases keep injected_faults=0 and reuse the states produced by the
reproduction experiments, regenerating them via the pinned venvs if absent).

Run::

    python exp_attest_eval_baseline.py --out artifacts/attest_eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DS_WORKDIR = os.path.join(HERE, "_ckpt", "zero_to_fp32_repro")
SW_WORKDIR = os.path.join(HERE, "_ckpt", "swiglu_reshard_repro")
STRICT_RTOL = 1e-3
REALISTIC_RTOL = 1e-2


def _ensure(workdir: str, probe_file: str, exp_script: str) -> None:
    if os.path.exists(os.path.join(workdir, probe_file)):
        return
    env = {**os.environ, "PYTHONPATH": HERE}
    r = subprocess.run([sys.executable, os.path.join(HERE, exp_script),
                        "--workdir", workdir], env=env, cwd=HERE,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{exp_script} failed while regenerating states:\n"
                           f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def _max_rel_dev(y, y_ref) -> float:
    import torch

    if not torch.isfinite(y).all():  # a NaN/inf output trips any loss check
        return float("inf")
    return float((y - y_ref).abs().max() / (y_ref.abs().max() + 1e-30))


def _certify(pre_states, post_states, *, optim_pre=None, optim_post=None,
             progress_pre=None, progress_post=None):
    from attest.gate import certify_transition
    from attest.snapshot import snapshot_from_state_dicts

    pre = snapshot_from_state_dicts(pre_states, optim_pre, progress=progress_pre)
    post = snapshot_from_state_dicts(post_states, optim_post, progress=progress_post)
    return certify_transition(pre, post)


def _case_record(name, kind, corpus_class, dev, decision, expect_commit,
                 expect_forward_blind=None) -> dict:
    rec = {
        "name": name,
        "kind": kind,
        "corpus_class": corpus_class,
        "output_max_rel_dev": dev,
        "eval_catches_strict": dev > STRICT_RTOL,
        "eval_catches_realistic": dev > REALISTIC_RTOL,
        "certificate_committed": decision.committed,
        "violations_by_invariant": sorted({v.invariant for v in decision.violations}),
        "expected_committed": expect_commit,
        "as_expected": decision.committed == expect_commit,
    }
    if expect_forward_blind is not None:
        rec["expected_forward_blind"] = expect_forward_blind
        rec["as_expected"] = rec["as_expected"] and (
            (dev == 0.0) == expect_forward_blind
        )
    return rec


# --------------------------------------------------------------------------- #
# Natural case 1: the zero_to_fp32 conversions (DeepSpeed#6791).
# --------------------------------------------------------------------------- #
def _natural_ds_cases() -> list:
    import torch

    import exp_attest_zero_to_fp32_repro as ds_repro

    _ensure(DS_WORKDIR, os.path.join("ckpt", "live_truth.pt"),
            "exp_attest_zero_to_fp32_repro.py")
    truth = torch.load(os.path.join(DS_WORKDIR, "ckpt", "live_truth.pt"),
                       map_location="cpu", weights_only=False)
    model = ds_repro._build_model()
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(32, 64, generator=g)

    def forward(state):
        model.load_state_dict(state)
        with torch.no_grad():
            return model(x)

    y_ref = forward(truth)
    out = []
    for arm, expect_commit in (("buggy_sharded", False), ("buggy_default", False),
                               ("fixed", True)):
        converted = ds_repro._load_converted(os.path.join(DS_WORKDIR, f"out_{arm}"))
        dev = _max_rel_dev(forward(converted), y_ref)
        out.append(_case_record(
            f"zero_to_fp32_{arm}", "natural", "DeepSpeed#6791 conversion corruption",
            dev, _certify(truth, converted), expect_commit))
    return out


# --------------------------------------------------------------------------- #
# Natural case 2: the SwiGLU TP=2 -> TP=1 reshard (Megatron-LM PR#520).
# --------------------------------------------------------------------------- #
def _natural_swiglu_cases() -> list:
    import torch
    import torch.nn.functional as torch_fn

    _ensure(SW_WORKDIR, "truth_buggy.pt", "exp_attest_swiglu_reshard_repro.py")
    truth = torch.load(os.path.join(SW_WORKDIR, "truth_buggy.pt"), weights_only=True)
    ffn = truth["mlp.linear_fc1.weight"].shape[0] // 2
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(16, truth["mlp.linear_fc1.weight"].shape[1], generator=g)

    def forward(state):
        fc1, fc2 = state["mlp.linear_fc1.weight"], state["mlp.linear_fc2.weight"]
        gate, up = x @ fc1[:ffn].T, x @ fc1[ffn:].T
        return (torch_fn.silu(gate) * up) @ fc2.T

    y_ref = forward(truth)
    out = []
    for arm, expect_commit in (("buggy", False), ("fixed", True)):
        loaded = torch.load(os.path.join(SW_WORKDIR, f"loaded_{arm}_tp1.pt"),
                            weights_only=True)
        dev = _max_rel_dev(forward(loaded), y_ref)
        out.append(_case_record(
            f"swiglu_reshard_{arm}", "natural", "Megatron-LM PR#520 SwiGLU mis-split",
            dev, _certify(truth, loaded), expect_commit))
    return out


# --------------------------------------------------------------------------- #
# Designed cases: documented bug classes the eval oracle is blind (or nearly
# blind) to, injected via the corpus-mapped fault library on real Adam state.
# --------------------------------------------------------------------------- #
DESIGNED = [
    # (fault name, corpus class, expect_forward_blind)
    ("stale_optimizer_moment", "Megatron-LM#761 optimizer merge/reshard class", True),
    ("step_counter_reset", "bias-correction restart class", True),
    ("progress_reset", "trainer-progress metadata loss", True),
    ("precision_cast", "narrow-dtype transport round-trip (PR#2789-adjacent)", None),
    ("value_corruption", "mis-stitched shard boundary / partial write", None),
]


def _designed_cases() -> list:
    import torch

    import exp_attest_zero_to_fp32_repro as ds_repro
    from attest.faults import get_fault

    torch.manual_seed(99)
    model = ds_repro._build_model(seed=99)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        loss = model(torch.randn(8, 64)).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

    fqns = [n for n, _ in model.named_parameters()]
    raw = optim.state_dict()["state"]
    model_ref = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optim_ref = {fqns[i]: {s: (v.clone() if isinstance(v, torch.Tensor) else v)
                           for s, v in raw[i].items()} for i in raw}
    progress_ref = {"global_step": 3, "samples_seen": 24}

    g = torch.Generator().manual_seed(1234)
    x = torch.randn(32, 64, generator=g)

    def forward(state):
        model.load_state_dict(state)
        with torch.no_grad():
            return model(x)

    y_ref = forward(model_ref)
    out = []
    for fault_name, corpus_class, expect_blind in DESIGNED:
        model_sd = {k: v.clone() for k, v in model_ref.items()}
        optim_sd = {"state": {f: {s: (v.clone() if isinstance(v, torch.Tensor) else v)
                                  for s, v in slots.items()}
                              for f, slots in optim_ref.items()}}
        progress = dict(progress_ref)
        note = get_fault(fault_name)(model_sd, optim_sd, progress, {})
        dev = _max_rel_dev(forward(model_sd), y_ref)
        decision = _certify(model_ref, model_sd,
                            optim_pre={"state": optim_ref}, optim_post=optim_sd,
                            progress_pre=progress_ref, progress_post=progress)
        rec = _case_record(fault_name, "designed", corpus_class, dev, decision,
                           expect_commit=False, expect_forward_blind=expect_blind)
        rec["fault_note"] = note
        out.append(rec)
    return out


def run(args: argparse.Namespace) -> int:
    global DS_WORKDIR, SW_WORKDIR
    DS_WORKDIR = os.path.abspath(args.ds_workdir)
    SW_WORKDIR = os.path.abspath(args.sw_workdir)
    cases = _natural_ds_cases() + _natural_swiglu_cases() + _designed_cases()
    natural_buggy = [c for c in cases
                     if c["kind"] == "natural" and not c["expected_committed"]]
    blind = [c for c in cases if c["output_max_rel_dev"] == 0.0
             and not c["expected_committed"]]
    report = {
        "exp": "attest-eval-baseline",
        "eval_oracle": {
            "definition": "same fixed batch, bit-exact reference output captured at "
                          "save time, deterministic forward; catches iff max relative "
                          "output deviation exceeds the threshold",
            "strict_rtol": STRICT_RTOL,
            "realistic_rtol": REALISTIC_RTOL,
        },
        "cases": cases,
        "headline": {
            "natural_bugs_eval_also_catches_strict": all(
                c["eval_catches_strict"] for c in natural_buggy),
            "forward_blind_corrupting_cases": [c["name"] for c in blind],
            "forward_blind_all_caught_by_certificate": all(
                not c["certificate_committed"] for c in blind),
            "all_as_expected": all(c["as_expected"] for c in cases),
        },
    }
    print("=" * 78)
    print(f"{'case':26s} {'kind':9s} {'rel dev':>10s}  eval@1e-3  eval@1e-2  certificate")
    for c in cases:
        verdict = "COMMIT" if c["certificate_committed"] else "ABORT"
        print(f"{c['name']:26s} {c['kind']:9s} {c['output_max_rel_dev']:10.3g}  "
              f"{str(c['eval_catches_strict']):>9s}  {str(c['eval_catches_realistic']):>9s}  "
              f"{verdict} {c['violations_by_invariant']}")
    h = report["headline"]
    print(f"natural bugs also caught by the strict eval oracle: "
          f"{h['natural_bugs_eval_also_catches_strict']}")
    print(f"forward-blind corrupting cases: {h['forward_blind_corrupting_cases']} "
          f"-> all caught by certificate: {h['forward_blind_all_caught_by_certificate']}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")
    return 0 if h["all_as_expected"] else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None)
    p.add_argument("--ds-workdir", default=DS_WORKDIR,
                   help="zero_to_fp32 repro state dir (regenerated if missing)")
    p.add_argument("--sw-workdir", default=SW_WORKDIR,
                   help="SwiGLU reshard repro state dir (regenerated if missing)")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    raise SystemExit(main())
