# DCP load silently casts checkpoint tensors to the template dtype,
# truncating precision with no warning (size mismatches raise loudly).
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
print(f"max |saved - loaded|: {dev:.6f}  <- silent precision loss")
