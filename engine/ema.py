from __future__ import annotations

import torch
from torch import nn

@torch.no_grad()
def ema_update(model: nn.Module, model_ema: nn.Module, decay: float) -> None:
    for p, p_ema in zip(model.parameters(), model_ema.parameters()):
        p_ema.copy_(p_ema * decay + p.detach() * (1.0 - decay))
    for b, b_ema in zip(model.buffers(), model_ema.buffers()):
        b_ema.copy_(b_ema * decay + b.detach() * (1.0 - decay))


def compute_ema_decay(iters: int, warmup_iters: int = 200, max_decay: float = 0.996) -> float:
    iters = int(iters)
    warmup_iters = int(warmup_iters)
    return float(min(iters / (iters + warmup_iters), max_decay))

