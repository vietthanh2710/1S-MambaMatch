from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(p))


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location)


def maybe_remove(path: Optional[str | Path]) -> None:
    if not path:
        return
    p = Path(path)
    if p.exists():
        p.unlink()

