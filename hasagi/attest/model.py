"""A self-contained, deterministically-initialized GPT-style model.

No external weights or network access: the architecture mirrors GPT-2 (token +
position embeddings, pre-norm transformer blocks, weight-tied LM head) and is
parameterized so the same module runs as a fast test model ("tiny") or at the
GPT-2-small scale ("gpt2_125m"). Determinism comes from seeding the generator
explicitly at construction, so every rank builds bit-identical weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    d_model: int
    max_seq: int
    # GPT-2 ties the LM head to the token embedding; tying is kept opt-in here
    # because tied FQNs surface differently across wrappers, which would test
    # the wrapper's aliasing rules rather than the reshard transport.
    tie_weights: bool = False

    @property
    def name(self) -> str:
        return f"gpt-d{self.d_model}-l{self.n_layer}-h{self.n_head}-v{self.vocab_size}"


PRESETS = {
    # fast unit-test scale (536,064 params)
    "tiny": GPTConfig(vocab_size=512, n_layer=2, n_head=2, d_model=128, max_seq=64),
    # mid scale for the transition matrix (13,439,232 params)
    "small": GPTConfig(vocab_size=8192, n_layer=4, n_head=4, d_model=384, max_seq=128),
    # GPT-2-small shape (162,447,360 params: GPT-2's 124M ties lm_head to wte;
    # this model keeps them separate, so the 50257x768 block is counted twice)
    "gpt2_125m": GPTConfig(vocab_size=50257, n_layer=12, n_head=12, d_model=768, max_seq=256),
}


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, *, seed: int = 1234) -> None:
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(seed)  # every rank builds identical weights
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, s = idx.shape
        mask = torch.triu(torch.full((s, s), float("-inf"), device=idx.device), diagonal=1)
        x = self.tok(idx) + self.pos(torch.arange(s, device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

    def loss(self, idx: torch.Tensor) -> torch.Tensor:
        logits = self.forward(idx[:, :-1])
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), idx[:, 1:].reshape(-1)
        )


def synthetic_batch(cfg: GPTConfig, batch: int, *, seed: int) -> torch.Tensor:
    """A fixed-seed token batch so every run sees identical data."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (batch, cfg.max_seq), generator=g)
