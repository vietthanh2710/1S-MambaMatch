from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms


def normalize_pil(img: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(img)


def blur(img: Image.Image, p: float = 0.5) -> Image.Image:
    if random.random() < p:
        sigma = float(np.random.uniform(0.1, 2.0))
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def crop(img: Image.Image, mask: Image.Image, size: int, ignore_value: int = 255) -> Tuple[Image.Image, Image.Image]:
    """
    Crop the image and mask to the given size.
    """
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask

def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.5609, 0.5440, 0.5332], [0.2935, 0.2962, 0.3017]),
    ])(img)

    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask

def obtain_cutmix_box(
    img_hw: Tuple[int, int],
    p: float = 0.5,
    size_min: float = 0.02,
    size_max: float = 0.4,
    ratio_1: float = 0.3,
    ratio_2: float = 1 / 0.3,
) -> torch.Tensor:
    """
    Returns (H,W) mask with ones in a random rectangle.
    Notebook used (H,W) ordering; we keep that.
    """
    h, w = int(img_hw[0]), int(img_hw[1])
    mask = torch.zeros((h, w), dtype=torch.float32)
    if random.random() > p:
        return mask

    area = float(np.random.uniform(size_min, size_max) * h * w)
    for _ in range(50):
        ratio = float(np.random.uniform(ratio_1, ratio_2))
        cut_w = int(np.sqrt(area / ratio))
        cut_h = int(np.sqrt(area * ratio))
        x = int(np.random.randint(0, w))
        y = int(np.random.randint(0, h))
        if x + cut_w <= w and y + cut_h <= h:
            mask[y : y + cut_h, x : x + cut_w] = 1.0
            return mask
    return mask

