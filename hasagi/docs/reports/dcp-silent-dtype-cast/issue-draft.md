# [DCP] load silently casts checkpoint tensors to the template dtype, truncating precision without any warning

Filed as <https://github.com/pytorch/pytorch/issues/187138> (2026-06-12).

## 🐛 Describe the bug

`torch.distributed.checkpoint.load` resolves what to read from the
user-provided `state_dict` template. When a template tensor's dtype differs
from the dtype stored in the checkpoint, the load completes with no warning
or error and the values are silently cast to the template dtype. If the
template dtype is narrower than the checkpoint dtype (for example a
`bfloat16` tensor receiving a `float32` checkpoint value), the restored
state loses precision relative to what was saved, and nothing at any layer
reports it.

The same code path treats a shape mismatch as a hard error, so the silence
is asymmetric: sizes are checked, dtypes are not.

```python
import tempfile

import torch
import torch.distributed.checkpoint as dcp

torch.manual_seed(0)
ckpt = tempfile.mkdtemp()
saved = {"w": torch.rand(4, 3, dtype=torch.float32) * 100}
dcp.save(state_dict=saved, checkpoint_id=ckpt)

template = {"w": torch.zeros(4, 3, dtype=torch.bfloat16)}  # wrong dtype
dcp.load(state_dict=template, checkpoint_id=ckpt)  # no warning, no error

dev = (saved["w"].double() - template["w"].double()).abs().max()
print(f"loaded dtype: {template['w'].dtype} (checkpoint stored float32)")
print(f"max |saved - loaded|: {dev:.6f}")
```

Output (identical on every version listed under Versions below):

```
loaded dtype: torch.bfloat16 (checkpoint stored float32)
max |saved - loaded|: 0.177818
```

## Why this matters in practice

The dangerous direction is the narrowing one. A mixed-precision trainer
that keeps fp32 master weights (or fp32 Adam moments) and resumes through a
template built from a bf16 model restores silently truncated masters: the
job trains on, the loss curve looks plausible, and the high-precision state
the optimizer depended on is gone. Framework history shows this exact class
shipping as silent correctness bugs (for example Megatron-LM PR#2789,
where bf16 working copies desynced from fp32 masters across a load).

## Where it happens

In the filesystem reader's load path, the size mismatch is an explicit
assertion but the dtype is never compared before the copy
(`torch/distributed/checkpoint/filesystem.py`, `_load_tensors` inner
function):

```python
if target_tensor.size() != tensor.size():
    raise AssertionError(
        f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
    )
target_tensor.copy_(tensor)   # Tensor.copy_ casts dtypes silently
```

The planner already knows the stored dtype
(`item.tensor_data.properties.dtype` in the metadata), so detecting the
mismatch costs nothing at plan time.

## Expected behavior

Either of these would make the behavior safe:

1. raise on dtype mismatch by default, mirroring the size check, with an
   explicit opt-in (planner flag) for intentional load-time casting; or
2. keep the cast but emit a clear warning naming the tensor, the checkpoint
   dtype, and the template dtype.

Neither the `dcp.load` docstring nor `DefaultLoadPlanner`'s documentation
mentions dtype conversion today, so current behavior is also undocumented.

## Versions

Reproduced identically on:

- torch 2.5.1
- torch 2.12.0 (current stable)
- torch 2.13.0.dev20260611 (nightly)

(CPU, single process, `no_dist` path; the copy site is in the shared
filesystem reader, so the distributed path goes through the same code.)
