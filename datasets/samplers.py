from __future__ import annotations

import math
import random
from typing import Iterator, Optional

import torch
from torch.utils.data import Sampler


class FewShotBatchSampler(Sampler[list[int]]):
    """
    Builds few-shot episodes by sampling:
      - `shot_num` support indices
      - `query_num` query indices

    Then yields a single batch of indices: [support..., query...]
    """

    def __init__(
        self,
        dataset_size: int,
        shot_num: int,
        query_num: int,
        episodes_per_epoch: int,
        *,
        type_data: str = "train",
        nsample: Optional[int] = None,
    ):
        self.dataset_size = int(dataset_size)
        self.shot_num = int(shot_num)
        self.query_num = int(query_num)
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.type_data = str(type_data)

        if self.query_num <= 0:
            raise ValueError("query_num must be > 0")
        if self.shot_num < 0:
            raise ValueError("shot_num must be >= 0")

        if nsample is not None and nsample > self.dataset_size:
            nsample = int(nsample)
            self.num_batches_per_epoch = nsample // self.query_num
            indices = torch.arange(self.dataset_size)
            times = math.ceil(nsample / len(indices))
            self.indices = indices.repeat(times)[:nsample]
        else:
            self.num_batches_per_epoch = self.dataset_size // self.query_num
            self.indices = torch.arange(self.dataset_size)

    def __iter__(self) -> Iterator[list[int]]:
        if self.type_data in {"train_l", "train_u", "train"}:
            shuffled = self.indices[torch.randperm(len(self.indices))]
        else:
            shuffled = self.indices

        current = 0
        for _ in range(self.episodes_per_epoch):
            query = shuffled[current : current + self.query_num]
            current += self.query_num

            if self.shot_num > 0:
                available = shuffled[~torch.isin(shuffled, query)]
                support = random.sample(list(available.cpu().numpy()), self.shot_num)
                support = torch.tensor(support, dtype=torch.int64)
                batch = torch.cat([support, query]).tolist()
            else:
                batch = query.tolist()

            yield batch

    def __len__(self) -> int:
        return int(self.num_batches_per_epoch)

