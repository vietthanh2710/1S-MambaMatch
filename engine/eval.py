from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from utils.meters import AverageMeter
from utils.metrics import SegmentationMetrics


@torch.no_grad()
def evaluate_fewshot(model, loader: DataLoader, metrictor: SegmentationMetrics, device: torch.device) -> float:
    model.eval()
    dice_meter = AverageMeter()

    for batch in loader:
        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        support_image, query_image = images[:1], images[1:]
        support_mask, query_mask = masks[:1], masks[1:]
        support_mask = torch.cat((1 - support_mask, support_mask), dim=1).float()

        pred, _ = model(query_image, support_image, support_mask, comp_drop=False)
        _, dice, _, _ = metrictor(query_mask.squeeze(1).cpu(), pred.cpu())
        dice_meter.update(float(dice))

    return float(dice_meter.avg)

