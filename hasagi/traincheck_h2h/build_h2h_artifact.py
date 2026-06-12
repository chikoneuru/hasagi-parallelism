"""Assemble the TrainCheck head-to-head artifact from the run directories.

Reads the checker result folders produced by the protocol in README.md,
applies the pre-registered scoring (a catch needs a violation unique to the
corrupted arm versus its matched controls), and writes one JSON artifact.
The triage notes are filled from the measured side experiments and are part
of the record, not editorial.
"""

import argparse
import glob
import json
import os


def failed_descriptions(check_dir: str) -> list:
    out = []
    for f in glob.glob(os.path.join(check_dir, "traincheck_checker_results_*",
                                    "*", "failed.log")):
        depth, buf = 0, []
        for line in open(f):
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0 and any(c.strip() for c in buf):
                try:
                    out.append(json.loads("".join(buf))["invariant"]["text_description"])
                except Exception:
                    pass
                buf = []
    return sorted(out)


def summary_counts(check_dir: str) -> dict:
    log = os.path.join(check_dir, "check.log")
    counts = {}
    if os.path.exists(log):
        for line in open(log):
            if "Total failed invariants:" in line:
                counts["failed_count"] = line.split(":")[1].strip()
            if "not triggered:" in line:
                counts["not_triggered_count"] = line.split(":")[-1].strip()
    return counts


def arm(check_dir: str) -> dict:
    return {"failed": failed_descriptions(check_dir), **summary_counts(check_dir)}


def score(corrupt: dict, controls: list) -> dict:
    control_union = set()
    for c in controls:
        control_union |= set(c["failed"])
    unique = sorted(set(corrupt["failed"]) - control_union)
    return {"unique_violations_vs_controls": unique,
            "raw_verdict": "flagged" if unique else "miss"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scen-a", required=True)
    p.add_argument("--scen-b", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    a = {name: arm(os.path.join(args.scen_a, f"check_{name}"))
         for name in ("A1_converter", "A1b", "A2_corrupted", "A3_control",
                      "A4_control2", "A5_healthy_diff")}
    b = {name: arm(os.path.join(args.scen_b, f"check_{name}"))
         for name in ("B1_buggy_save", "B1ref_save_control", "B2_buggy",
                      "B3_charity", "B4_control", "B5_healthy_alt",
                      "B6_buggy_sametp")}

    report = {
        "exp": "attest-traincheck-h2h",
        "tool": {"name": "traincheck", "version": "0.1.2",
                 "torch": "2.12.0+cu130, CPU execution (outside the tool's tested "
                          "1.7-2.5 envelope: megatron-core 0.17.1 forces the torch "
                          "upgrade; the published tutorial positive control was "
                          "re-run on this exact stack and detects its documented "
                          "bug, 123/989 violations incl. the optimizer "
                          "step/zero_grad root-cause relations)"},
        "scoring": "catch = violation unique to the corrupted arm vs matched "
                   "controls, within the first corrupted-state-consuming "
                   "iteration, surviving triage; reference traces never "
                   "double as check arms",
        "scenario_a": {
            "bug": "DeepSpeed#6791 (offline zero_to_fp32 conversion)",
            "arms": a,
            "converter_arm": score(a["A1_converter"], [a["A1b"]]),
            "consumer_arm": score(a["A2_corrupted"],
                                  [a["A3_control"], a["A4_control2"],
                                   a["A5_healthy_diff"]]),
            "triage": {
                "converter": "buggy and fixed converter runs produce identical "
                             "failed sets (environment-mismatch noise only); "
                             "the corrupting conversion is indistinguishable",
                "consumer": "the unique load-time violation tracks a metadata "
                            "side-channel: the buggy converter emits tensors "
                            "with requires_grad=False (0/6) where healthy "
                            "outputs have 6/6; robust across same-file, "
                            "second-seed, and different-healthy-values "
                            "controls",
            },
        },
        "scenario_b": {
            "bug": "Megatron-LM PR#520 (TP=2 save -> TP=1 load SwiGLU mis-split)",
            "arms": b,
            "save_arm": score(b["B1_buggy_save"], [b["B1ref_save_control"]]),
            "consumer_arm": score(b["B2_buggy"],
                                  [b["B4_control"], b["B5_healthy_alt"]]),
            "charity_arm": score(b["B3_charity"],
                                 [b["B4_control"], b["B5_healthy_alt"],
                                  b["B1ref_save_control"]]),
            "consumer_arm_after_b6_triage": score(b["B2_buggy"],
                                                  [b["B4_control"],
                                                   b["B5_healthy_alt"],
                                                   b["B6_buggy_sametp"]]),
            "triage": "every violation on the corrupted TP-change arm also "
                      "fires on the same-TP control whose values reconstruct "
                      "bit-exactly (B2's full set is contained in B6's): the "
                      "signal tracks the pre-fix declaration's code path "
                      "(import/registration order relations), not the value "
                      "corruption, and would flag correct loads identically",
        },
        "positive_control": {
            "what": "published 5-minute tutorial: invariants from mnist.py, "
                    "detection on the 84911 buggy pipeline, on this exact stack",
            "result": "123/989 invariants violated, including the "
                      "Adadelta.step / Optimizer.zero_grad relations the "
                      "tutorial documents as the root-cause signal",
        },
        "collector_adaptations": [
            "lazy CUDA attribute probes shielded (host driver/torch-2.5 stub)",
            "torch.distributed.checkpoint/_shard wrapped without argument "
            "dumps (dumper cannot serialize sharded-tensor objects)",
            "typename() made exception-safe for __torch_function__ guards",
            "megatron consumer arms use the sampler model tracker (the proxy "
            "tracker fails megatron-core's strict module type checks)",
            "save jobs traced API-only (the sampler tracker requires an "
            "optimizer, which a pure checkpoint job does not have)",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    for scen, key in (("scenario_a", "consumer_arm"), ("scenario_b", "consumer_arm")):
        print(scen, key, report[scen][key]["raw_verdict"],
              report[scen][key]["unique_violations_vs_controls"])
    print("scenario_a converter_arm", report["scenario_a"]["converter_arm"]["raw_verdict"])
    print("scenario_b save_arm", report["scenario_b"]["save_arm"]["raw_verdict"])
    print("scenario_b charity_arm", report["scenario_b"]["charity_arm"]["raw_verdict"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
