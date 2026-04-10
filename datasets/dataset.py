from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from datasets.augmentations import blur, normalize_pil, obtain_cutmix_box


class _RandomCropPair(transforms.RandomResizedCrop):
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        out = []
        for img in imgs:
            out.append(TF.resized_crop(img, i, j, h, w, self.size, self.interpolation))
        return out

@dataclass(frozen=True)
class MedicalImageSplit:
    x_train_l: np.ndarray
    y_train_l: np.ndarray
    x_train_u: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class MedicalImageDataset(Dataset):
    """
    Dataset wrapper matching the notebook's behavior.

    type_data:
      - train_l: returns (img_tensor, mask_tensor[1,H,W] long)
      - train_u: returns (weak_tensor, strong1_tensor, strong2_tensor, ignore_mask[H,W] long, cutmix1[H,W], cutmix2[H,W])
      - test:    returns (img_tensor, mask_tensor[1,H,W] long)
    """

    def __init__(
        self,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None,
        type_data: Literal["train_l", "train_u", "test"] = "train_l",
        augment: bool = True,
        crop_size: Tuple[int, int] = (192, 256),
    ):
        self.images = images
        self.masks = masks
        self.type_data = type_data
        self.augment_enabled = bool(augment) and type_data in ("train_l", "train_u")
        self.crop_size = crop_size

        self._color = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
        self._gray = transforms.RandomGrayscale(p=0.2)

    def __len__(self) -> int:
        return int(len(self.images))

    def _rotate(self, image: Image.Image, mask: Image.Image, degrees=(-10, 10), p=0.5):
        if torch.rand(1).item() < p:
            degree = float(np.random.uniform(*degrees))
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask

    def _hflip(self, image: Image.Image, mask: Image.Image, p=0.5):
        if torch.rand(1).item() < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def _vflip(self, image: Image.Image, mask: Image.Image, p=0.5):
        if torch.rand(1).item() < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def _random_resized_crop(self, image: Image.Image, mask: Image.Image, p=0.1):
        if torch.rand(1).item() < p:
            image, mask = _RandomCropPair(self.crop_size, scale=(0.8, 0.95))([image, mask])
        return image, mask

    def _augment_pair(self, image: Image.Image, mask: Image.Image):
        image, mask = self._random_resized_crop(image, mask)
        image, mask = self._rotate(image, mask)
        image, mask = self._hflip(image, mask)
        image, mask = self._vflip(image, mask)
        return image, mask

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[idx])

        if self.type_data != "train_u":
            if self.masks is None:
                raise ValueError("masks are required for non-train_u")
            mask = Image.fromarray(self.masks[idx])
        else:
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))

        if self.augment_enabled:
            image, mask = self._augment_pair(image, mask)

        if self.type_data != "train_u":
            img_t = transforms.ToTensor()(image)
            m = np.asarray(mask, np.int64)
            mask_t = torch.from_numpy(m[np.newaxis])  # (1,H,W)
            return img_t, mask_t

        # Unlabeled: produce weak + 2 strong with cutmix boxes, like notebook
        image_w, image_s1, image_s2 = deepcopy(image), deepcopy(image), deepcopy(image)

        if random.random() < 0.8:
            image_s1 = self._color(image_s1)
        image_s1 = self._gray(image_s1)
        image_s1 = blur(image_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box((image_s1.size[1], image_s1.size[0]), p=0.5)

        if random.random() < 0.8:
            image_s2 = self._color(image_s2)
        image_s2 = self._gray(image_s2)
        image_s2 = blur(image_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box((image_s2.size[1], image_s2.size[0]), p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0]), dtype=np.uint8))

        img_w_t = normalize_pil(image_w)
        img_s1_t = normalize_pil(image_s1)
        img_s2_t = normalize_pil(image_s2)
        ignore_t = torch.from_numpy(np.array(ignore_mask)).long()

        # keep notebook behavior for 254 -> ignore 255 mapping (even though train_u mask is zeros)
        mask_arr = torch.from_numpy(np.array(mask)).long()
        ignore_t[mask_arr == 254] = 255

        return img_w_t, img_s1_t, img_s2_t, ignore_t, cutmix_box1, cutmix_box2

