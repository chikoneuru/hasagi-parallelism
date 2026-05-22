"""Disk-backed checkpoint store with partition-aware metadata.

When a job is sharded across N workers and later resumes on N' workers, the resume
code needs to know which layer ranges each checkpoint file holds — see Hydrozoa
MLSys'22 §3.5 for the original heterogeneous-resharding pattern HISE inherits.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckpointMeta:
    job_id: str
    step: int
    shard_id: int
    layer_range: tuple[int, int]   # [start, end] layer indices inclusive
    path: str


@dataclass
class CheckpointStore:
    root: Path
    meta_index: list[CheckpointMeta] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_shard(self, meta: CheckpointMeta, payload: bytes) -> None:
        target = self.root / f"{meta.job_id}-{meta.step:08d}-{meta.shard_id:02d}.bin"
        target.write_bytes(payload)
        meta.path = str(target)
        self.meta_index.append(meta)
        (self.root / "index.json").write_text(
            json.dumps([m.__dict__ for m in self.meta_index], indent=2)
        )

    def shards_for(self, job_id: str, step: int) -> Iterable[CheckpointMeta]:
        return (m for m in self.meta_index if m.job_id == job_id and m.step == step)
