from __future__ import annotations

import torch
import torch.nn.functional as F

def ASS_loss(
    s_logits: torch.Tensor,
    w_logits: torch.Tensor,
    *,
    tau: float = 0.4,
) -> torch.Tensor:
    """
    Adaptive Self-Supervised Loss:

    - `w_logits`: teacher (EMA) logits on weakly augmented images
    - `s_logits`: student logits on strongly augmented images

    We compute uncertainty from teacher entropy and down-weight CE where uncertain.
    """
    w_prob = torch.softmax(w_logits, dim=1)
    entropy_w = -torch.sum(w_prob * torch.log(w_prob + 1e-6), dim=1)  # (B,H,W)

    uncertain = (entropy_w > tau).float()
    adjust = torch.ones_like(uncertain)
    adjust[uncertain == 1] = 0.3  # matches notebook

    pseudo = torch.argmax(w_logits, dim=1)  # (B,H,W)

    ce = F.cross_entropy(s_logits, pseudo, reduction="none")  # (B,H,W)
    s_prob = torch.softmax(s_logits, dim=1)
    entropy_s = -torch.sum(s_prob * torch.log(s_prob + 1e-6), dim=1)  # (B,H,W)

    loss = (ce * adjust + entropy_s * uncertain).mean()
    return loss

