# Documented reshard/checkpoint-transition bugs (evidence corpus)

A corpus of naturally-occurring, documented bugs in the checkpoint
save/load/reshard/conversion paths of production training systems. Each entry
maps the bug to the certificate invariant that would catch it at the
transition gate (`attest/certificate.py`), states whether loss-curve-only
validation would have missed it, and sketches how its root-cause class is
re-injected in the CPU harness (`attest/faults.py`).

Verification status: every entry below was independently re-checked against
the cited URL (the issue/PR/document exists, the title matches, and the root
cause matches the page); of 46 collected entries, 45 passed and 1 was rejected
(see the final section). 38 entries are bugs; 7 are validation-practice
records quoting how elastic-training systems actually validate reshard
correctness.

Provenance note. The frequently-quoted "11 silent Megatron bugs" figure comes
from TTrace (arXiv:2506.09280, Table 1): those are predominantly *runtime
parallelism* bugs (wrong gradient synchronization under TP/SP/CP), not
checkpoint/reshard bugs, and should not be cited as reshard evidence. Only the
state-corrupting members of that set that manifest across a transition
boundary are included here (replica divergence #599/#1446). The
checkpoint-path entries below come from direct issue-tracker mining.

## Verified entries (38 bugs)

### Megatron-LM and the Megatron-DeepSpeed/BLOOM lineage

| ID | What goes wrong | Silent | Invariant that catches it |
|---|---|---|---|
| [#1570](https://github.com/NVIDIA/Megatron-LM/issues/1570) | torch_dist resume diverges (mcore-0.12); writer sporadically emits corrupted tensors (~1e-42 denormals); torch-format resume is exact, torch_dist is not | yes | content_equivalence (read-back of written shards pre-commit) |
| [#761](https://github.com/NVIDIA/Megatron-LM/issues/761) | MoE (EP>1) resume mis-loads dual-optimizer state between expert/non-expert optimizer instances; maintainer-confirmed, fixed in [1505db4](https://github.com/NVIDIA/Megatron-LM/commit/1505db4cc4e9e94ee22583c76f7e425ea34f5aea) | yes | optimizer_accounting |
| [PR#520](https://github.com/NVIDIA/Megatron-LM/pull/520) | SwiGLU fused gate/up weight declared as one contiguous TP-sharded tensor; loading at a different TP degree re-splits at wrong boundaries, interleaving gate/up rows. Residency and shapes PASS; only content comparison catches it. **REPRODUCED + CAUGHT BLIND through the real pipeline (2026-06-11)**: pre-fix declaration vendored verbatim from the PR's base commit (`ab0336a`), saved at TP=2 and reloaded at TP=1 via real megatron-core 0.17.1 `dist_checkpointing` (CPU/gloo); loaded fc1 is bit-exactly the predicted `[gate0;up0;gate1;up1]` mis-split, every value preserved as a multiset (L2/numel checksums blind — norm is permutation-invariant), certificate aborts on the single fingerprint violation, zero injected faults; same-TP roundtrip and the shipped `apply_swiglu_sharded_factory` both commit bit-exact (max dev 0.0) — the bug fires exclusively on the reshard transition. See `exp_attest_swiglu_reshard_repro.py`, `tests/test_swiglu_reshard_repro.py`, `artifacts/attest_swiglu_reshard_repro.json` | yes | content_equivalence |
| [PR#1770](https://github.com/NVIDIA/Megatron-LM/pull/1770) | Wrong `replica_id` ownership declaration for ShardedTensors under expert-tensor-parallelism: redundant replicas treated as distinct owners | yes | param_residency (ownership census from metadata) |
| [PR#2789](https://github.com/NVIDIA/Megatron-LM/pull/2789) | BF16 + precision-aware optimizer + CPU offload: load restores subtly wrong model params (bf16 working copies desync from fp32 masters) | yes | content_equivalence (+ master/working consistency) |
| [PR#2658](https://github.com/NVIDIA/Megatron-LM/pull/2658) | Dist-ckpt RNG sharding keyed only on TP+PP; EP ranks restore wrong generator streams | yes | **NONE — gap** (see below) |
| [PR#4828](https://github.com/NVIDIA/Megatron-LM/pull/4828) | Context-parallel RNG tracker restores wrong dropout RNG state across save/load | yes | **NONE — gap** (see below) |
| [#599](https://github.com/NVIDIA/Megatron-LM/issues/599) | SwitchMLP router weights never synchronized within the TP group; nominally-replicated copies diverge from step 0 (fixed by PR#619) | yes | content_equivalence (replica consistency at the transition) |
| [#1446](https://github.com/NVIDIA/Megatron-LM/issues/1446) | TP+SP final_layernorm gradients not reduced across the TP group; replicated layernorm replicas diverge (fixed by PR#1528) | yes | content_equivalence (replica consistency) |
| [#656](https://github.com/NVIDIA/Megatron-LM/issues/656) | Embedding/LM-head tied weights silently untie under the distributed optimizer with overlapped param gather; the two views of one logical block receive different updates (fixed in db2040f) | yes | content_equivalence over declared aliases |
| [BLOOM/DeepSpeed-1801](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) | BLOOM-176B: BF16Optimizer (DeepSpeed PR#1801) skipped gradient clipping on TP ranks > 0; layernorm replicas diverged for ~10 days and were discovered only at a TP=4 to TP=1 checkpoint merge — the flagship case of corruption exposed exactly at a reshard transition (TrainCheck's motivating example) | yes | content_equivalence (replica consistency at the transition) |

### DeepSpeed

| ID | What goes wrong | Silent | Invariant that catches it |
|---|---|---|---|
| [#6771](https://github.com/microsoft/DeepSpeed/issues/6771) | Params frozen before `deepspeed.initialize` (ZeRO-2): optimizer-state coverage in the checkpoint disagrees with the model's parameter set; the issue itself requests a save-time consistency validation | yes | optimizer_accounting |
| [#4272](https://github.com/deepspeedai/DeepSpeed/issues/4272) | BF16_Optimizer checkpoint save/load crashes under ZeRO-1 (`fp16_groups` attribute assumption; `load_serial` kwarg mismatch); fixed in PR#4434 | no (loud) | optimizer_accounting; loud control case for abort-path testing |
| [#6691](https://github.com/deepspeedai/DeepSpeed/issues/6691) | Universal-checkpoint conversion of a ZeRO-3 checkpoint runs without error but the resumed run behaves as if from scratch: structure-valid, content-wrong | yes | content_equivalence |
| [#7546](https://github.com/deepspeedai/DeepSpeed/issues/7546) | With `load_universal=true`, ZeRO-3 save writes model-state shards only on rank 0; other ranks' shards are silently never written | yes | param_residency (over the saved artifact) |
| [#7584](https://github.com/deepspeedai/DeepSpeed/issues/7584) | `ds_to_universal.py` treats all ZeRO-3 subgroups as one flat buffer; offset arithmetic mis-slices optimizer state with multiple subgroups; fixed in PR#7585 | partially | content_equivalence + optimizer_accounting (element-count conservation) |
| [PR#7599](https://github.com/deepspeedai/DeepSpeed/pull/7599) | DP expansion (2→4): new ranks demand rank-indexed files the smaller run never produced; loader indexes old-layout artifacts by new-layout rank ids; fixed 2025-10-01 | no (loud; silent variant exists with stale same-named files) | param_residency (block→source-shard mapping) |
| [#6791](https://github.com/deepspeedai/DeepSpeed/issues/6791) | `zero_to_fp32.py` with `max_shard_size`: shard writer mutates the in-memory consolidated state dict, emitting all-zero weights (v0.16.0, fixed in PR#6792). **REPRODUCED + CAUGHT BLIND through the real pipeline (2026-06-11)**: real ZeRO-2 CPU checkpoint converted by the checkpoint's own v0.16.0 script; DEFAULT CLI flags corrupt too (`max_shard_size` defaults to 5GB, so every standard conversion hits the bug; zero-frac ~0.8 plus uninitialized-memory garbage, exit 0); certificate aborts with 12 content_equivalence violations, zero injected faults; v0.16.1 commits bit-exact (max dev 0.0). See `exp_attest_zero_to_fp32_repro.py`, `tests/test_zero_to_fp32_repro.py`, `artifacts/attest_zero_to_fp32_repro.json` | yes | content_equivalence (post-write read-back) |
| [#5489](https://github.com/deepspeedai/DeepSpeed/issues/5489) | With bf16 enabled, frozen parameters never enter the BF16_Optimizer groups and are silently omitted from saved checkpoints (`exclude_frozen_parameters` notwithstanding) | yes | param_residency |
| [#1896](https://github.com/deepspeedai/DeepSpeed/issues/1896) | `zero_to_fp32.py` silently drops tied parameters (GPT-2 lm_head tied to wte): aliased FQNs get no entry in the converted state dict (fixed in PR#3033 via data_ptr alias matching) | yes | param_residency |
| [#3824](https://github.com/deepspeedai/DeepSpeed/issues/3824) | ZeRO-2 variant of the tied-weight hole in `get_fp32_state_dict_from_zero_checkpoint`; hit PyTorch Lightning users (fixed in PR#3825) | yes | param_residency |

### PyTorch DCP / FSDP state-dict paths

The most directly relevant slice: our harness uses these exact APIs on CPU, so
a certificate detection here is an end-to-end story with no GPU.

| ID | What goes wrong | Silent | Invariant that catches it |
|---|---|---|---|
| [#144657](https://github.com/pytorch/pytorch/issues/144657) | `async_save` with CPU tensors aliases live training state (`.to(cpu)` is a no-op on CPU tensors): the checkpoint silently contains values from LATER steps | yes | content_equivalence |
| [#140898](https://github.com/pytorch/pytorch/issues/140898) | `get_optimizer_state_dict` silently returns empty optimizer state when params still carry grads (`_init_optim_state` early-returns); round-trips as `state: {}` with no warning | yes | optimizer_accounting |
| [#164929](https://github.com/pytorch/pytorch/issues/164929) | `get_optimizer_state_dict` mutates the live optimizer (hidden lr=0 step advances Adam step counters): a "read-only" snapshot changes the training trajectory | yes | optimizer_accounting |
| [#126285](https://github.com/pytorch/pytorch/issues/126285) | `StateDictOptions(strict=True)` ignored by `set_model_state_dict` under full_state_dict/broadcast modes: missing keys silently keep stale weights | yes | content_equivalence |
| [#117421](https://github.com/pytorch/pytorch/issues/117421) | FSDP silently discards `load_state_dict` issued between forward and backward (writes lost when the flat param reshards) | yes | content_equivalence |
| [#140900](https://github.com/pytorch/pytorch/issues/140900) | DCP round-trip drops `initial_lr` from optimizer param_groups, silently breaking LR-scheduler resume | yes | optimizer_accounting |
| [#143828](https://github.com/pytorch/pytorch/issues/143828) | `set_optimizer_state_dict` corrupts param_groups when an empty param group exists; malformed groups surface only at the next `step()` | no (delayed crash) | optimizer_accounting |
| [#102821](https://github.com/pytorch/pytorch/issues/102821) | World-size reshard (8 to 64 nodes) of FSDP sharded optimizer state produces empty/None local shards; `load_sharded_optimizer_state_dict` crashes | no (loud) | param_residency |
| [#126881](https://github.com/pytorch/pytorch/issues/126881) | `dcp.load` is in-place: checkpoint keys absent from the local state dict are skipped with no warning (the umbrella semantics behind several silent-load classes) | yes | content_equivalence |
| [#92823](https://github.com/pytorch/pytorch/issues/92823) | `load_sharded_optimizer_state_dict` returns param_groups as an unhydrated BytesIO placeholder when `flatten_sharded_tensors=True` | yes | optimizer_accounting |

### Elastic-training systems and HF Trainer

| ID | What goes wrong | Silent | Invariant that catches it |
|---|---|---|---|
| [Tenplex#32](https://github.com/kungfu-team/tenplex/pull/32) | State transformer mis-parses Megatron optimizer state for fp32 checkpoints (hard-coded fp16-wrapper key path): param_groups silently mishandled during redistribution | yes | optimizer_accounting |
| [Oobleck#16](https://github.com/SymbioticLab/Oobleck/issues/16) | Microbatch-redistribution ILP returns None during reconfiguration when the solver leaves a pipeline's count unset; pipeline re-instantiation crashes | no (loud) | microbatch_invariant |
| [torchtitan#811](https://github.com/pytorch/torchtitan/issues/811) | Restart at a larger world size fails: per-rank dataloader/LR-scheduler state (`dataloader.dp_rank_N`) is not reshardable; new ranks request keys that do not exist | no (loud; silent shrink variant exists) | param_residency (key coverage); microbatch_invariant |
| [torchtitan#409](https://github.com/pytorch/torchtitan/issues/409) | Resume after a dp/tp-degree change: dataloader state for new ranks is absent; training continues on a fresh data stream (warning only), replaying/skipping data | yes | microbatch_invariant |
| [transformers#43708](https://github.com/huggingface/transformers/issues/43708) | Resume with a changed per-device batch size silently restores the stale `train_batch_size`, corrupting max_steps and the LR schedule | yes | microbatch_invariant; optimizer_accounting |
| [transformers#38939](https://github.com/huggingface/transformers/issues/38939) | Resume re-initializes the within-epoch step counter to -1: the final gradient-accumulation window is consumed but never applied | yes | microbatch_invariant |
| [transformers#39215](https://github.com/huggingface/transformers/issues/39215) | Resume loads RNG state AFTER fetching the first batch: bit-reproducible resume breaks whenever data loading is stochastic | yes | NONE unless RNG/stream state is in scope — third instance of the auxiliary-state gap |

## The auxiliary-state invariant gap (research finding)

Three verified bugs corrupt **auxiliary per-rank stream state** (RNG trackers,
dataloader stream position) across a save/load/reshard boundary: Megatron
PR#2658 and PR#4828 (RNG streams mis-keyed under EP/CP), and HF
transformers#39215 (RNG restored after the first batch fetch on resume). A
wrong-but-internally-valid stream passes all four state invariants: it is not
a parameter, not an optimizer slot, not a progress counter, and not a
reduction order. The torchtitan entries (#811, #409) show the same class at
the dataloader-cursor level. This motivates a fifth invariant,
*auxiliary-state residency*: every named auxiliary stream (RNG generator
state, dataloader cursor) has exactly one owner across the transition and
survives bit-exact, unless its re-seeding is declared.

## Validation-practice evidence (why a transition gate is needed; all verified)

- DynaTrain (arXiv:2605.18815, May 2026) validates resharding correctness
  exclusively by comparing loss-convergence trajectories against a static
  baseline (quoted in its correctness section). No public repo or issue
  tracker as of 2026-06-10.
- ElasWave (arXiv:2510.00606v3) says "parameters are verified, redistributed,
  and loaded on-the-fly", but the "verification" is snapshot-based recovery
  mechanics; correctness is evaluated post-run via loss deviation against a
  no-failure baseline (its section 7.5). No pre-commit invariant check; no
  public repo as of 2026-06-10.
- Tenplex (SOSP'24, arXiv:2312.05181v3 section 6.8) validates by
  loss-over-steps overlay ("the loss does not diverge when the resources
  increase/decrease"); the in-repo convergence harness is an MNIST-scale CNN
  (PR#40/PR#49), and the state transformer ships no equivalence tests, which
  is consistent with Tenplex#32 above surviving into the fp32 path.
- Oobleck (SOSP'23, arXiv:2309.08125) has no correctness evaluation of
  reconfiguration in the paper (throughput-only; correctness asserted
  procedurally); reconfiguration logic is covered by unit tests (PR#9), and
  the reconfiguration-path bug Oobleck#16 above shipped despite them.
- ByteCheckpoint ships no published reshard-equivalence test; community
  report [#7](https://github.com/ByteDance-Seed/ByteCheckpoint/issues/7)
  (4-GPU FSDP checkpoint loaded on 1 GPU crashes) was resolved with usage
  guidance, not a checker. UCP and ByteCheckpoint validate by loss-curve
  convergence and unit tests per their papers; DeepSpeed #6691/#7546 above
  are field reports of UCP conversions that pass silently.
- Bamboo's public repo is research-artifact grade: three issues total, no
  test suite for reconfiguration consistency.
- TorchElastic documentation positions restart-from-checkpoint as the entire
  correctness story for membership changes, which makes the checkpoint
  reshard/restore path (DCP; pytorch#92823 above) the single point of silent
  failure.

## Rejected after verification (1 of 46)

- pytorch#145378 ("Loading weights using torch.distributed.checkpoint leads
  to large loss values"): the issue is real and silent, but it was closed
  2025-04-29 with the root cause identified as a `to_empty()` buffer-wipe
  during FSDP meta-device materialization, explicitly exonerating `dcp.load`.
  Not usable as a DCP-substrate bug; excluded.

## Fault-injector mapping (`attest/faults.py`)

| Injector | Documented class it emulates |
|---|---|
| silent_unloaded_param | strict=False unloaded-key class (pytorch#126285, #126881, #117421) |
| value_corruption | Megatron#1570 corrupted-shard writes; pytorch#144657 aliased async save |
| permuted_values | Megatron PR#520 chunk-then-shard mis-split |
| cross_param_overwrite | converter key mis-mapping class; tied-alias holes (DeepSpeed#1896, #3824, Megatron#656) |
| stale_optimizer_moment | Megatron#761 / DeepSpeed#6771 mis-load; pytorch#140898 empty-state round-trip |
| step_counter_reset | bias-correction restart on resume; pytorch#164929 hidden lr=0 step |
| precision_cast | Megatron PR#2789 precision desync |
| progress_reset | trainer-progress loss at the boundary (transformers#38939, torchtitan#409) |
| dropped_fqn | DeepSpeed#7546 never-written shard; torchtitan#811 missing per-rank keys |
| reduction_order_swap | undeclared reduction-order change (drift unbounded) |
