from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from segmentation_models_pytorch.losses import DiceLoss
from torch.utils.data import DataLoader

from datasets.dataset import MedicalImageDataset
from datasets.samplers import FewShotBatchSampler
from engine.eval import evaluate_fewshot
from engine.train import TrainState, save_best_checkpoints
from models import OneShotMambaUnet
from utils.io import ensure_dir, load_yaml
from utils.metrics import SegmentationMetrics
from utils.seed import set_seed

# -------------------------
# Args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def train_one_epoch_sup(
    model,
    trainloader,
    optimizer,
    criterion_l1,
    criterion_l2,
    device,
):
    model.train()

    total_loss = 0.0

    for batch in trainloader:
        img_q, mask_q, img_s, mask_s = batch  # few-shot format

        img_q = img_q.to(device)
        mask_q = mask_q.to(device)
        img_s = img_s.to(device)
        mask_s = mask_s.to(device)

        logits, _ = model(img_q, img_s, mask_s)

        loss = criterion_l1(logits, mask_q) + criterion_l2(logits, mask_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(trainloader)


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 142))
    set_seed(seed)

    device_str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)

    # -------------------------
    # Data
    # -------------------------
    data_cfg = cfg["data"]
    npz_path = Path(data_cfg["npz_path"])

    data = np.load(str(npz_path))
    x, y = data["image"], data["mask"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=int(data_cfg.get("test_size", 520)),
        random_state=seed
    )

    traindataset = MedicalImageDataset(x_train, y_train, type_data="train")
    testdataset = MedicalImageDataset(x_test, y_test, type_data="test", augment=False)

    few = cfg["fewshot"]
    shot = int(few.get("shot_num", 1))
    q = int(few.get("query_num", 4))

    trainsampler = FewShotBatchSampler(
        len(traindataset),
        shot_num=shot,
        query_num=q,
        episodes_per_epoch=len(traindataset) // q,
        type_data="train",
    )

    testsampler = FewShotBatchSampler(
        len(testdataset),
        shot_num=shot,
        query_num=1,
        episodes_per_epoch=len(testdataset),
        type_data="test",
    )

    num_workers = int(data_cfg.get("num_workers", 4))

    trainloader = DataLoader(
        traindataset,
        batch_sampler=trainsampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    testloader = DataLoader(
        testdataset,
        batch_sampler=testsampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = OneShotMambaUnet(n_channels=3, n_classes=2)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    train_cfg = cfg["train"]
    lr = float(train_cfg.get("lr", 5e-5))
    wd = float(train_cfg.get("weight_decay", 0.01))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=wd
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(train_cfg.get("factor", 0.6)),
        patience=int(train_cfg.get("patience", 9)),
    )


    criterion_l1 = DiceLoss(mode="multiclass")
    criterion_l2 = nn.CrossEntropyLoss()
    metrictor = SegmentationMetrics(average=True, ignore_background=True)

    out_cfg = cfg.get("output", {})
    out_dir = ensure_dir(out_cfg.get("dir", "outputs"))
    save_prefix = str(out_cfg.get("save_prefix", "supervised"))

    state = TrainState()
    epochs = int(train_cfg.get("epochs", 150))

    # Training Loop
    for epoch in range(epochs):

        train_loss = train_one_epoch_sup(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            criterion_l1=criterion_l1,
            criterion_l2=criterion_l2,
            device=device,
        )

        dice = evaluate_fewshot(model, testloader, metrictor, device=device)

        scheduler.step(dice)

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | dice={dice:.4f}"
        )

        save_best_checkpoints(
            out_dir=Path(out_dir),
            save_prefix=save_prefix,
            model=model,
            model_ema=None,   # ❌ no EMA
            optimizer=optimizer,
            state=state,
            epoch=epoch,
            dice=dice,
            dice_ema=None,
        )


if __name__ == "__main__":
    main()