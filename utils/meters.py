from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class AverageMeter:
    window: int = 0
    _history: List[float] = field(default_factory=list, init=False)
    _sum: float = 0.0
    _count: int = 0
    val: float = 0.0
    avg: float = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._sum = 0.0
        self._count = 0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        if self.window > 0:
            if n != 1:
                raise ValueError("windowed AverageMeter only supports n=1")
            self._history.append(float(val))
            if len(self._history) > self.window:
                self._history.pop(0)
            self.val = self._history[-1]
            self.avg = float(np.mean(self._history)) if self._history else 0.0
        else:
            self.val = float(val)
            self._sum += float(val) * int(n)
            self._count += int(n)
            self.avg = self._sum / max(1, self._count)

