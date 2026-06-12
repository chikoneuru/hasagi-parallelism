# TrainCheck head-to-head on the two reproduced natural bugs

Executed comparison between TrainCheck (OSDI'25, `pip install traincheck`)
and the transition certificate, on the two natural bugs reproduced by
`exp_attest_zero_to_fp32_repro.py` (DeepSpeed #6791) and
`exp_attest_swiglu_reshard_repro.py` (Megatron-LM PR#520). The result
artifact is `artifacts/attest_traincheck_h2h.json`, built by
`build_h2h_artifact.py` from the run directories.

## Environment

One dedicated venv (gitignored). A torch 2.5.0 pin (TrainCheck's tested
envelope tops out at 2.5) was attempted first, but megatron-core 0.17.1
requires a newer torch and the resolver upgrades it: the runs use torch
2.12.0+cu130, OUTSIDE the tool's tested 1.7--2.5 envelope. The published
5-minute tutorial (mnist reference -> infer -> 84911 buggy pipeline) was
re-run on this exact stack as the positive control and detects the
documented bug (123/989 violations, including the optimizer step/zero_grad
relations the tutorial names as the root-cause signal), validating
collector, inference, and checker end to end. CPU execution throughout
(the cu130 wheel cannot initialize against this host's driver).

```bash
python3 -m venv .venv/traincheck
.venv/traincheck/bin/pip install traincheck deepspeed==0.16.0 megatron-core==0.17.1 \
    huggingface_hub safetensors ninja psutil efficientnet_pytorch torchvision
# host shims (see sitecustomize_cuda_shield.py for why each exists):
SP=.venv/traincheck/lib/python3.12/site-packages
cp traincheck_h2h/sitecustomize_cuda_shield.py $SP/tc_cuda_shield.py
echo "import tc_cuda_shield" > $SP/zz_tc_cuda_shield.pth
```

The shims make the COLLECTOR survive on this host; none alters invariant
semantics: (1) lazy CUDA attribute probes raise against this host's driver,
so the cuFFT plan cache returns AttributeError instead; (2) torch's
distributed-checkpoint plumbing passes sharded-tensor objects whose
arguments the dumper cannot serialize, so those APIs are wrapped without
argument dumps; (3) `typename()` falls back to the plain class name when an
object's `__torch_function__` guard rejects introspection; (4) dispatch and
registration plumbing (`torch.utils._pytree`, `torch._ops`,
`torch._library`, `torch.export`, and the distributed-checkpoint internals)
is excluded from instrumentation entirely, applied symmetrically to every
arm and reference: with it instrumented, the pre-fix declaration's load
path alone emitted a 10 GB trace from a 144-parameter model and exceeded
40 minutes before timing out.

## Protocol (pre-registered scoring)

Catch = at least one violation in `failed.log` that fires no later than the
first iteration consuming the corrupted state AND does not also fire on the
matched healthy control arm, surviving the manual triage TrainCheck's own
evaluation performs. Miss = empty failed set or only violations shared with
controls. Reference traces for inference never double as check arms.

Positive control: the published 5-minute tutorial (invariants from
`mnist.py`, detection on the 84911 buggy pipeline) is run first to validate
the installation end to end.

## Scenario A: offline conversion (DeepSpeed #6791)

All commands run from a scratch dir with `CUDA_VISIBLE_DEVICES=`,
`DS_ACCELERATOR=cpu`, and the venv's bin on PATH. `$TC` is the collector.

1. Checkpoint (untraced): `python traincheck_h2h/ds_train.py` with
   `TC_H2H_CKPT=<dir>` (single-process real ZeRO-2, 30 steps; DeepSpeed
   copies `zero_to_fp32.py` into the checkpoint).
2. Conversions (untraced): the checkpoint's own 0.16.0 script with DEFAULT
   flags -> `out_buggy`; the 0.16.1 script -> `out_fixed`; a second
   checkpoint (different data seed) + 0.16.1 -> `out_fixed2`.
3. Reference traces: `$TC -p ds_train.py --models-to-track model` and
   `$TC -p consumer.py --models-to-track model` on `out_fixed` (seed 1).
4. Infer: `traincheck-infer -f trace_ref_train trace_ref_consumer`.
5. Arms (each `$TC` then `traincheck-check`):
   - A1  converter itself, buggy script (via `convert_traced.py`)
   - A1b converter itself, fixed script (matched control)
   - A2  consumer on `out_buggy` (seed 2)
   - A3  consumer on `out_fixed` (seed 2, control)
   - A4  consumer on `out_fixed` (seed 3, second control)
   - A5  consumer on `out_fixed2` (healthy-but-different-values control)

## Scenario B: TP-degree-change reshard (Megatron-LM PR#520)

Same environment. The model proxy tracker breaks megatron-core's strict
module checks, so consumer arms use `--model-tracker-style sampler`; the
save job has no optimizer, which the sampler requires, so save traces are
API-only.

1. Checkpoints (untraced, torchrun 2 ranks): `save_tp2.py` with
   `TC_H2H_ARM` buggy/fixed -> `ckpt_buggy`, `ckpt_fixed`; plus
   `TC_H2H_PAINT_BASE=1100` fixed -> `ckpt_fixed_alt`.
2. Reference traces: traced save (fixed arm, via `run_save_tp2.sh`
   shscript) and traced load consumer on `ckpt_fixed` (data seed 301).
3. Infer from both reference traces.
4. Arms:
   - B1  traced save, buggy declaration (vs B1b, a second traced FIXED save,
     so the inference reference never doubles as a control)
   - B2  load consumer on `ckpt_buggy` (data seed 302)
   - B4  load consumer on `ckpt_fixed` (data seed 302, control)
   - B5  load consumer on `ckpt_fixed_alt` (healthy-different-values control)
   - B3  maximal charity: checker given BOTH the buggy save trace and the
     buggy load trace in one invocation
   - B6  the decisive triage control: the buggy declaration loaded at the
     SAME TP degree (torchrun 2 ranks via run_load_tp2.sh), where the values
     reconstruct bit-exactly. Every violation the corrupted arm raises must
     be checked against this arm: a signal that also fires when the values
     are correct tracks the declaration's code path, not the corruption.

## Building the artifact

```bash
python traincheck_h2h/build_h2h_artifact.py \
    --scen-a _traincheck/h2h/scenA --scen-b _traincheck/h2h/scenB \
    --out artifacts/attest_traincheck_h2h.json
```
