# Documented reshard/checkpoint-transition bugs (evidence corpus)

A corpus of naturally-occurring, documented bugs in the checkpoint
save/load/reshard/conversion paths of production training systems. Each entry
maps the bug to the certificate invariant that would catch it at the
transition gate (`attest/certificate.py`), states whether loss-curve-only
validation would have missed it, and sketches how its root-cause class is
re-injected in the CPU harness (`attest/faults.py`).

Verification status: **verified** entries were independently re-checked
against the cited URL (the issue/PR exists and the root cause matches);
**unverified** entries were collected the same way but their independent
re-check has not run yet — do not cite them in a paper until verified.

Provenance note. The frequently-quoted "11 silent Megatron bugs" figure comes
from TTrace (arXiv:2506.09280, Table 1): those are predominantly *runtime
parallelism* bugs (wrong gradient synchronization under TP/SP/CP), not
checkpoint/reshard bugs, and should not be cited as reshard evidence. Only the
state-corrupting members of that set that manifest across a transition
boundary are included here (replica divergence #599/#1446). The
checkpoint-path entries below come from direct issue-tracker mining.

## Verified entries (16)

### Megatron-LM

| ID | What goes wrong | Silent | Invariant that catches it |
|---|---|---|---|
| [#1570](https://github.com/NVIDIA/Megatron-LM/issues/1570) | torch_dist resume diverges (mcore-0.12); writer sporadically emits corrupted tensors (~1e-42 denormals); torch-format resume is exact, torch_dist is not | yes | content_equivalence (read-back of written shards pre-commit) |
| [#761](https://github.com/NVIDIA/Megatron-LM/issues/761) | MoE (EP>1) resume mis-loads dual-optimizer state between expert/non-expert optimizer instances; maintainer-confirmed, fixed in [1505db4](https://github.com/NVIDIA/Megatron-LM/commit/1505db4cc4e9e94ee22583c76f7e425ea34f5aea) | yes | optimizer_accounting |
| [PR#520](https://github.com/NVIDIA/Megatron-LM/pull/520) | SwiGLU fused gate/up weight declared as one contiguous TP-sharded tensor; loading at a different TP degree re-splits at wrong boundaries, interleaving gate/up rows. Residency and shapes PASS; only content comparison catches it | yes | content_equivalence |
| [PR#1770](https://github.com/NVIDIA/Megatron-LM/pull/1770) | Wrong `replica_id` ownership declaration for ShardedTensors under expert-tensor-parallelism: redundant replicas treated as distinct owners | yes | param_residency (ownership census from metadata) |
| [PR#2789](https://github.com/NVIDIA/Megatron-LM/pull/2789) | BF16 + precision-aware optimizer + CPU offload: load restores subtly wrong model params (bf16 working copies desync from fp32 masters) | yes | content_equivalence (+ master/working consistency) |
| [PR#2658](https://github.com/NVIDIA/Megatron-LM/pull/2658) | Dist-ckpt RNG sharding keyed only on TP+PP; EP ranks restore wrong generator streams | yes | **NONE — gap** (see below) |
| [PR#4828](https://github.com/NVIDIA/Megatron-LM/pull/4828) | Context-parallel RNG tracker restores wrong dropout RNG state across save/load | yes | **NONE — gap** (see below) |
| [#599](https://github.com/NVIDIA/Megatron-LM/issues/599) | SwitchMLP router weights never synchronized within the TP group; nominally-replicated copies diverge from step 0 (fixed by PR#619) | yes | content_equivalence (replica consistency at the transition) |
| [#1446](https://github.com/NVIDIA/Megatron-LM/issues/1446) | TP+SP final_layernorm gradients not reduced across the TP group; replicated layernorm replicas diverge (fixed by PR#1528) | yes | content_equivalence (replica consistency) |

### DeepSpeed

| ID | What goes wrong | Silent | Invariant that catches it |
|---|---|---|---|
| [#6771](https://github.com/microsoft/DeepSpeed/issues/6771) | Params frozen before `deepspeed.initialize` (ZeRO-2): optimizer-state coverage in the checkpoint disagrees with the model's parameter set; the issue itself requests a save-time consistency validation | yes | optimizer_accounting |
| [#4272](https://github.com/deepspeedai/DeepSpeed/issues/4272) | BF16_Optimizer checkpoint save/load crashes under ZeRO-1 (`fp16_groups` attribute assumption; `load_serial` kwarg mismatch); fixed in PR#4434 | no (loud) | optimizer_accounting; loud control case for abort-path testing |
| [#6691](https://github.com/deepspeedai/DeepSpeed/issues/6691) | Universal-checkpoint conversion of a ZeRO-3 checkpoint runs without error but the resumed run behaves as if from scratch: structure-valid, content-wrong | yes | content_equivalence |
| [#7546](https://github.com/deepspeedai/DeepSpeed/issues/7546) | With `load_universal=true`, ZeRO-3 save writes model-state shards only on rank 0; other ranks' shards are silently never written | yes | param_residency (over the saved artifact) |
| [#7584](https://github.com/deepspeedai/DeepSpeed/issues/7584) | `ds_to_universal.py` treats all ZeRO-3 subgroups as one flat buffer; offset arithmetic mis-slices optimizer state with multiple subgroups; fixed in PR#7585 | partially | content_equivalence + optimizer_accounting (element-count conservation) |
| [PR#7599](https://github.com/deepspeedai/DeepSpeed/pull/7599) | DP expansion (2→4): new ranks demand rank-indexed files the smaller run never produced; loader indexes old-layout artifacts by new-layout rank ids; fixed 2025-10-01 | no (loud; silent variant exists with stale same-named files) | param_residency (block→source-shard mapping) |
| [#6791](https://github.com/deepspeedai/DeepSpeed/issues/6791) | `zero_to_fp32.py` with `max_shard_size`: shard writer mutates the in-memory consolidated state dict, emitting all-zero weights (v0.16.0, fixed in PR#6792) | yes | content_equivalence (post-write read-back) |

## The two invariant gaps (research finding)

Two verified, recent, maintainer-acknowledged Megatron bugs (PR#2658, PR#4828)
corrupt **auxiliary per-rank stream state** (RNG trackers) across a
save/load/reshard boundary. A wrong-but-internally-valid RNG stream passes all
four certificate invariants: it is not a parameter, not an optimizer slot, not
a progress counter, and not a reduction order. This motivates a fifth
invariant, *auxiliary-state residency*: every named auxiliary stream (RNG
generator state, dataloader cursor) has exactly one owner across the
transition and survives bit-exact, unless its re-seeding is declared.

## Validation-practice evidence (why a transition gate is needed)

- DynaTrain (arXiv:2605.18815, May 2026) validates resharding correctness
  exclusively by comparing loss-convergence trajectories against a static
  baseline (quoted in its correctness section).
- ElasWave (arXiv:2510.00606v3) says "parameters are verified, redistributed,
  and loaded on-the-fly", but the "verification" is snapshot-based recovery
  mechanics; correctness is evaluated post-run via loss deviation against a
  no-failure baseline (its §7.5). No pre-commit invariant check.
- UCP and ByteCheckpoint validate by loss-curve convergence and unit tests
  (per their papers); DeepSpeed #6691/#7546 above are field reports of UCP
  conversions that pass silently and corrupt state.

## Unverified entries (collected, independent re-check pending)

PyTorch/DCP and FSDP state-dict slice: pytorch#92823, #102821, #117421,
 #126285, #126881, #140898, #140900, #143828, #144657, #145378, #164929;
torchtitan#409, #811. Elastic systems: Tenplex#32, Oobleck#16. HF
transformers#38939, #39215, #43708. DeepSpeed#1896, #3824, #5489;
Megatron-LM#656; BigScience BLOOM/DeepSpeed-1801 (the TrainCheck motivating
case). Validation-practice quotes for Tenplex/ByteCheckpoint/Oobleck/Bamboo/
TorchElastic. These remain in the collection queue; verify each URL and root
cause before citing.

## Fault-injector mapping (`attest/faults.py`)

| Injector | Documented class it emulates |
|---|---|
| silent_unloaded_param | strict=False unloaded-key class (template values survive) |
| value_corruption | Megatron#1570 corrupted-shard writes |
| permuted_values | Megatron PR#520 chunk-then-shard mis-split |
| cross_param_overwrite | converter key mis-mapping class |
| stale_optimizer_moment | Megatron#761 / DeepSpeed#6771 optimizer-state mis-load |
| step_counter_reset | bias-correction restart on resume |
| precision_cast | Megatron PR#2789 precision desync |
| progress_reset | trainer-progress loss at the transition boundary |
| dropped_fqn | DeepSpeed#7546 never-written shard |
| reduction_order_swap | undeclared reduction-order change (drift unbounded) |
