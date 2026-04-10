from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from engine.checkpoint import maybe_remove, save_checkpoint
from engine.ema import compute_ema_decay, ema_update
from engine.losses import ASS_loss
from utils.meters import AverageMeter


@dataclass
class TrainState:
    epoch: int = -1
    iters: int = 0
    best_dice: float = 0.0
    best_dice_ema: float = 0.0
    best_epoch: int = 0
    best_epoch_ema: int = 0
    best_path: Optional[Path] = None
    best_path_ema: Optional[Path] = None


def train_one_epoch_semi(
    *,
    model: nn.Module,
    model_ema: nn.Module,
    trainloader_l: DataLoader,
    trainloader_u: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_l1,
    criterion_l2,
    device: torch.device,
    ema_warmup_iters: int,
    ema_max_decay: float,
    unsup_tau: float,
    loss_u_scale: float,
    state: TrainState,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = AverageMeter()
    total_loss_x = AverageMeter()
    total_loss_s = AverageMeter()

    loader = zip(trainloader_l, trainloader_u)
    for (img, mask), (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2) in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        img_u_w = img_u_w.to(device, non_blocking=True)
        img_u_s1 = img_u_s1.to(device, non_blocking=True)
        img_u_s2 = img_u_s2.to(device, non_blocking=True)
        ignore_mask = ignore_mask.to(device, non_blocking=True)
        cutmix_box1 = cutmix_box1.to(device, non_blocking=True)
        cutmix_box2 = cutmix_box2.to(device, non_blocking=True)

        img_x, mask_x = img[1:], mask[1:]
        img_sp, mask_sp1 = img[:1], mask[:1]
        mask_sp = torch.cat((1 - mask_sp1, mask_sp1), dim=1).float()

        with torch.no_grad():
            pred_u_w = model_ema(img_u_w, img_sp, mask_sp, comp_drop=False)[0].detach()
            mask_u_w = pred_u_w.argmax(dim=1)

        # Cutmix on images (as in notebook)
        box1 = cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1
        box2 = cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1
        img_u_s1[box1] = img_u_s1.flip(0)[box1]
        img_u_s2[box2] = img_u_s2.flip(0)[box2]

        pred_x, sp = model(img_x, img_sp, mask_sp)
        pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), sp, mask_sp, comp_drop=True).chunk(2)

        # Labeled loss
        loss_x = criterion_l1(pred_x, mask_x.squeeze(1)) + 0.5 * criterion_l2(pred_x, mask_x.squeeze(1))

        # Unsupervised loss (note: notebook had extra cutmix mask mixing; kept minimal and stable here)
        loss_u_s1 = unsupervised_loss(pred_u_s1, pred_u_w, tau=unsup_tau)
        loss_u_s2 = unsupervised_loss(pred_u_s2, pred_u_w, tau=unsup_tau)
        loss_u_s = (loss_u_s1 + loss_u_s2) * float(loss_u_scale)

        loss = loss_x + loss_u_s

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        total_loss_x.update(loss_x.item())
        total_loss_s.update(loss_u_s.item())

        state.iters += 1
        decay = compute_ema_decay(state.iters, warmup_iters=ema_warmup_iters, max_decay=ema_max_decay)
        ema_update(model, model_ema, decay=decay)

    return float(total_loss.avg), float(total_loss_x.avg), float(total_loss_s.avg)


def save_best_checkpoints(
    *,
    out_dir: Path,
    save_prefix: str,
    model,
    model_ema,
    optimizer,
    state: TrainState,
    epoch: int,
    dice: float,
    dice_ema: float,
) -> None:
    payload = {
        "model": model.state_dict(),
        "model_ema": model_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_dice": state.best_dice,
        "best_dice_ema": state.best_dice_ema,
        "best_epoch": state.best_epoch,
        "best_epoch_ema": state.best_epoch_ema,
    }

    # Always save last
    save_checkpoint(out_dir / "last.pth", payload)

    if dice >= state.best_dice:
        state.best_dice = float(dice)
        state.best_epoch = int(epoch)
        maybe_remove(state.best_path)
        state.best_path = out_dir / f"{save_prefix}-{state.best_dice:.4f}.pth"
        save_checkpoint(state.best_path, payload)

    if dice_ema >= state.best_dice_ema:
        state.best_dice_ema = float(dice_ema)
        state.best_epoch_ema = int(epoch)
        maybe_remove(state.best_path_ema)
        state.best_path_ema = out_dir / f"{save_prefix}-ema-{state.best_dice_ema:.4f}.pth"
        save_checkpoint(state.best_path_ema, payload)

