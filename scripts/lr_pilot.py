#!/usr/bin/env python3
"""Short controlled learning-rate pilot required by Phase 1."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from project_alpha.data import PackedTokenDataset
from project_alpha.model import AlphaLM, ModelConfig, count_parameters


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--model", choices=("reasoner", "baseline"), default="reasoner")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = json.loads((ROOT / "data" / "processed" / "metadata.json").read_text())
    dataset = PackedTokenDataset(ROOT / metadata["token_paths"]["train"], args.sequence_length)
    config = ModelConfig(vocab_size=int(metadata["tokenizer_vocab_size"]), n_stream_blocks=6 if args.model == "reasoner" else 7)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(dataset.n_examples, generator=generator)
    candidates = [2e-4, 3e-4, 5e-4, 6e-4]
    results = []
    for lr in candidates:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = AlphaLM(config, with_reasoner=args.model == "reasoner").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=torch.cuda.is_available())
        losses = []
        finite = True
        for step in range(args.steps):
            start = (step * args.batch_size) % max(1, len(indices) - args.batch_size)
            batch_indices = indices[start : start + args.batch_size].tolist()
            x, y = dataset.batch(batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, _ = model.training_loss(x, y, 2 if args.model == "reasoner" else 0)
            if not torch.isfinite(loss):
                finite = False
                break
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not math.isfinite(grad_norm):
                finite = False
                break
            optimizer.step()
            losses.append(float(loss.detach().float()))
        result = {"lr": lr, "finite": finite, "initial_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None, "steps": len(losses), "parameters": count_parameters(model)}
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    valid = [r for r in results if r["finite"] and r["final_loss"] is not None]
    selected = min(valid, key=lambda r: r["final_loss"])["lr"] if valid else None
    payload = {"candidates": results, "selected_lr": selected, "selection": "lowest final pilot loss among finite runs"}
    out = ROOT / "runs" / "lr_pilot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
