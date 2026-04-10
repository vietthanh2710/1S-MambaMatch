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
from engine.train import TrainState, save_best_checkpoints, train_one_epoch_semi
from models import OneShotMambaUnet
from utils.io import ensure_dir, load_yaml
from utils.metrics import SegmentationMetrics
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 142))
    set_seed(seed)

    device_str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)

    data_cfg = cfg["data"]
    npz_path = Path(data_cfg["npz_path"])
    data = np.load(str(npz_path))
    x, y = data["image"], data["mask"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=int(data_cfg.get("test_size", 520)), random_state=seed
    )

    labeled_count = int(data_cfg.get("labeled_count", 1037))
    x_train_u, x_train_l, _, y_train_l = train_test_split(x_train, y_train, test_size=labeled_count, random_state=seed)

    traindataset_l = MedicalImageDataset(x_train_l, y_train_l, type_data="train_l")
    traindataset_u = MedicalImageDataset(x_train_u, None, type_data="train_u")
    testdataset = MedicalImageDataset(x_test, y_test, type_data="test", augment=False)

    few = cfg["fewshot"]
    shot = int(few.get("shot_num", 1))
    q_l = int(few.get("query_num_l", 4))
    q_u = int(few.get("query_num_u", 4))

    episodes = len(traindataset_u) // q_u
    trainsampler_l = FewShotBatchSampler(
        len(traindataset_l), shot_num=shot, query_num=q_l, episodes_per_epoch=episodes, nsample=len(traindataset_u), type_data="train_l"
    )
    trainsampler_u = FewShotBatchSampler(
        len(traindataset_u), shot_num=0, query_num=q_u, episodes_per_epoch=episodes, type_data="train_u"
    )
    testsampler = FewShotBatchSampler(len(testdataset), shot_num=shot, query_num=1, episodes_per_epoch=len(testdataset), type_data="test")

    num_workers = int(data_cfg.get("num_workers", 4))
    trainloader_l = DataLoader(traindataset_l, batch_sampler=trainsampler_l, num_workers=num_workers, pin_memory=True)
    trainloader_u = DataLoader(traindataset_u, batch_sampler=trainsampler_u, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testdataset, batch_sampler=testsampler, num_workers=num_workers, pin_memory=True)

    model = OneShotMambaUnet(n_channels=3, n_classes=2)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    model_ema = OneShotMambaUnet(n_channels=3, n_classes=2)
    model_ema = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_ema)
    model_ema.load_state_dict(model.state_dict())
    model_ema.to(device)
    model_ema.eval()
    for p in model_ema.parameters():
        p.requires_grad = False

    train_cfg = cfg["train"]
    lr = float(train_cfg.get("lr", 5e-5))
    wd = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.Adam([{"params": [p for p in model.parameters()]}], lr=lr, betas=(0.9, 0.999), weight_decay=wd)

    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(sched_cfg.get("factor", 0.6)),
        patience=int(sched_cfg.get("patience", 9)),
    )

    criterion_l1 = DiceLoss(mode="multiclass")
    criterion_l2 = nn.CrossEntropyLoss()
    metrictor = SegmentationMetrics(average=True, ignore_background=True)

    ema_cfg = cfg.get("ema", {})
    loss_cfg = cfg.get("loss", {})
    warmup = int(ema_cfg.get("warmup_iters", 200))
    max_decay = float(ema_cfg.get("max_decay", 0.996))
    tau = float(loss_cfg.get("unsup_tau", 0.4))
    loss_u_scale = float(loss_cfg.get("loss_u_scale", 1.0 / 6.0))

    out_cfg = cfg.get("output", {})
    out_dir = ensure_dir(out_cfg.get("dir", "outputs"))
    save_prefix = str(out_cfg.get("save_prefix", "isic-50-1s"))

    state = TrainState()
    epochs = int(train_cfg.get("epochs", 150))

    for epoch in range(epochs):
        tl, tlx, tls = train_one_epoch_semi(
            model=model,
            model_ema=model_ema,
            trainloader_l=trainloader_l,
            trainloader_u=trainloader_u,
            optimizer=optimizer,
            criterion_l1=criterion_l1,
            criterion_l2=criterion_l2,
            device=device,
            ema_warmup_iters=warmup,
            ema_max_decay=max_decay,
            unsup_tau=tau,
            loss_u_scale=loss_u_scale,
            state=state,
        )

        dice = evaluate_fewshot(model, testloader, metrictor, device=device)
        dice_ema = evaluate_fewshot(model_ema, testloader, metrictor, device=device)
        scheduler.step(state.best_dice)

        print(
            f"Epoch {epoch:03d} | loss={tl:.4f} (x={tlx:.4f}, u={tls:.4f}) | dice={dice:.4f} | dice_ema={dice_ema:.4f}"
        )

        save_best_checkpoints(
            out_dir=Path(out_dir),
            save_prefix=save_prefix,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            state=state,
            epoch=epoch,
            dice=dice,
            dice_ema=dice_ema,
        )


if __name__ == "__main__":
    main()

