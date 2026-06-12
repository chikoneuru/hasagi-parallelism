"""Shared megatron-core SwiGLU pieces for the TrainCheck comparison runs.

Adapted from ``exp_attest_swiglu_reshard_repro.py`` (same geometry, painting,
and declarations) so the save/load scripts stay single-purpose and
TrainCheck's collector can launch them standalone. The pre-fix declaration is
the verbatim-geometry vendoring from NVIDIA/Megatron-LM commit ab0336a5 (the
base of fix PR#520); see the experiment file for the full provenance note.
"""

import os

import torch

HIDDEN = 6
FFN = 8
TP_SAVE = 2


def cpu_cuda_shim() -> None:
    """mcore's torch_dist checkpoint strategy calls torch.cuda unconditionally;
    on this host any lazy CUDA query raises, so no-op the two call sites."""
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.current_device = lambda: "cpu"


def init_parallel(tp: int) -> None:
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29578")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    cpu_cuda_shim()


def build_mlp(tp: int):
    import torch.nn.functional as torch_fn
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(7)
    cfg = TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=2, ffn_hidden_size=FFN,
        gated_linear_unit=True, activation_func=torch_fn.silu, add_bias_linear=False,
        use_cpu_initialization=True, tensor_model_parallel_size=tp,
    )
    return MLP(cfg, MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                  linear_fc2=RowParallelLinear))


def paint(mlp, rank: int, tp: int) -> None:
    base = float(os.environ.get("TC_H2H_PAINT_BASE", "1000"))
    half = FFN // tp
    with torch.no_grad():
        w1 = mlp.linear_fc1.weight
        ramp = torch.arange(HIDDEN, dtype=w1.dtype) * 1e-3
        for i in range(half):
            w1[i] = base + rank * half + i + ramp
            w1[half + i] = 2 * base + rank * half + i + ramp
        w2 = mlp.linear_fc2.weight
        ramp = torch.arange(HIDDEN, dtype=w2.dtype) * 1e-3
        for j in range(half):
            w2[:, j] = 3 * base + rank * half + j + ramp


def declare_pre_pr520(module, prefix: str = "") -> dict:
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing import ShardedTensor
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    state_dict = module.state_dict(prefix="mlp.", keep_vars=True)
    tensor_parallel_layers_axis_map = {
        'mlp.linear_fc1.weight': 0,
        'mlp.linear_fc1.bias': 0,
        'mlp.linear_fc2.weight': 1,
    }
    num_layers = 1
    global_layer_offset = 0
    sharded_state_dict = {}
    for layer_name in state_dict.keys():
        tensor = state_dict[layer_name]
        layer_key = f'{prefix}{global_layer_offset}.{layer_name}'
        sharded_offsets = [(0, global_layer_offset, num_layers)]
        if layer_name in tensor_parallel_layers_axis_map:
            tp_axis = tensor_parallel_layers_axis_map[layer_name]
            sharded_offsets.append(
                [tp_axis + 1,
                 parallel_state.get_tensor_model_parallel_rank(),
                 parallel_state.get_tensor_model_parallel_world_size()]
            )
            replica_id = parallel_state.get_data_parallel_rank()
        else:
            replica_id = (
                parallel_state.get_data_parallel_rank()
                * parallel_state.get_data_parallel_world_size()
                + parallel_state.get_tensor_model_parallel_rank()
            )
        if layer_name.endswith('._extra_state'):
            sharded_state_dict[layer_key] = ShardedObject(
                f'{prefix}{layer_name}', tensor, (num_layers,),
                (global_layer_offset,), replica_id,
            )
        else:
            sharded_state_dict[layer_key] = ShardedTensor.from_rank_offsets(
                f'{prefix}{layer_name}', tensor, *sharded_offsets,
                replica_id=replica_id, prepend_axis_num=1,
            )
    return sharded_state_dict


def declare_fixed(module) -> dict:
    from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

    return module.sharded_state_dict(prefix="mlp.",
                                     metadata=ensure_metadata_has_dp_cp_group(None))


def declare(module, arm: str) -> dict:
    return declare_pre_pr520(module) if arm == "buggy" else declare_fixed(module)
