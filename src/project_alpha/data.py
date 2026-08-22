"""Reproducible WikiText packing and deterministic batch access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


class PackedTokenDataset:
    def __init__(self, token_path: str | Path, sequence_length: int) -> None:
        self.token_path = Path(token_path)
        self.sequence_length = sequence_length
        self.tokens = np.load(self.token_path, mmap_mode="r")
        if self.tokens.dtype != np.uint32:
            raise ValueError(f"expected uint32 tokens, found {self.tokens.dtype}")
        self.n_examples = max(0, (len(self.tokens) - 1) // sequence_length)

    def batch(self, indices: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = np.stack(
            [self.tokens[int(i) * self.sequence_length : int(i + 1) * self.sequence_length] for i in indices]
        ).astype(np.int64, copy=False)
        targets = np.stack(
            [self.tokens[int(i) * self.sequence_length + 1 : int(i + 1) * self.sequence_length + 1] for i in indices]
        ).astype(np.int64, copy=False)
        return torch.from_numpy(inputs).to(device=device, non_blocking=True), torch.from_numpy(targets).to(
            device=device, non_blocking=True
        )

    def metadata(self) -> dict:
        return {
            "token_path": str(self.token_path),
            "token_count": int(len(self.tokens)),
            "sequence_length": self.sequence_length,
            "examples": self.n_examples,
        }


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
