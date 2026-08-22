#!/usr/bin/env python3
"""Train a resumable Phase-1 model or its plain Mamba-3 baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

# When this file is executed by path, Python puts ``scripts/`` before the
# standard library on sys.path.  The local profiling entry point is named
# profile.py, so leaving that path first makes cProfile import the wrong
# module during optimizer construction.
_SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _SCRIPT_DIR:
    sys.path.pop(0)

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from project_alpha.data import PackedTokenDataset
from project_alpha.model import AlphaLM, ModelConfig, count_parameters


ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("reasoner", "baseline"), default="reasoner")
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accumulation", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def make_config(vocab_size: int, model_kind: str) -> ModelConfig:
    # One additional stream block keeps the plain baseline within the same
    # 10--15M parameter budget as the reasoner model; exact counts are saved.
    return ModelConfig(vocab_size=vocab_size, n_stream_blocks=6 if model_kind == "reasoner" else 7)


def schedule_lambda(step: int, total_steps: int) -> float:
    warmup = max(1, int(round(0.015 * total_steps)))
    if step < warmup:
        return float(step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint_payload(
    model: AlphaLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    permutation: torch.Tensor,
    permutation_generator: torch.Generator,
    metrics: dict,
    depth_rng: random.Random,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": model.config.to_dict(),
        "with_reasoner": model.with_reasoner,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "permutation": permutation,
        "permutation_generator_state": permutation_generator.get_state(),
        "depth_rng_state": depth_rng.getstate(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = json.loads((ROOT / "data" / "processed" / "metadata.json").read_text())
    vocab_size = int(metadata["tokenizer_vocab_size"])
    config = make_config(vocab_size, args.model)
    model = AlphaLM(config, with_reasoner=args.model == "reasoner").to(device)
    dataset = PackedTokenDataset(ROOT / metadata["token_paths"]["train"], args.sequence_length)
    steps_per_epoch = math.ceil(dataset.n_examples / args.batch_size / args.grad_accumulation)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=torch.cuda.is_available()
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_lambda(step, total_steps))
    run_name = args.run_name or f"phase1_{args.model}"
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.jsonl"
    depth_rng = random.Random(args.seed + 7919)
    permutation_generator = torch.Generator().manual_seed(args.seed + 104729)
    permutation = torch.empty(0, dtype=torch.int64)
    epoch = 0
    batch_in_epoch = 0
    global_step = 0
    history: dict = {"best_loss": float("inf"), "steps": 0}

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
        batch_in_epoch = checkpoint["batch_in_epoch"]
        global_step = checkpoint["global_step"]
        permutation = checkpoint["permutation"]
        permutation_generator.set_state(checkpoint["permutation_generator_state"])
        depth_rng.setstate(checkpoint["depth_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        history = checkpoint.get("metrics", history)

    print(json.dumps({
        "device": str(device),
        "model": args.model,
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "dataset": dataset.metadata(),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "config": config.to_dict(),
    }, indent=2))

    start_time = time.perf_counter()
    while epoch < args.epochs:
        if permutation.numel() != dataset.n_examples:
            permutation = torch.randperm(dataset.n_examples, generator=permutation_generator)
            batch_in_epoch = 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        while batch_in_epoch * args.batch_size * args.grad_accumulation < dataset.n_examples:
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            depth = depth_rng.choice([1, 2, 4]) if args.model == "reasoner" else 0
            for micro in range(args.grad_accumulation):
                offset = (batch_in_epoch * args.grad_accumulation + micro) * args.batch_size
                if offset >= dataset.n_examples:
                    break
                indices = permutation[offset : min(offset + args.batch_size, dataset.n_examples)].tolist()
                inputs, targets = dataset.batch(indices, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss, depth_losses = model.training_loss(inputs, targets, depth)
                    scaled_loss = loss / args.grad_accumulation
                scaled_loss.backward()
                micro_losses.append(float(loss.detach().float()))
            grad_norm = float(clip_grad_norm_(model.parameters(), 1.0))
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {global_step}: {grad_norm}")
            optimizer.step()
            scheduler.step()
            batch_in_epoch += 1
            global_step += 1
            history["steps"] = global_step
            mean_loss = float(sum(micro_losses) / max(1, len(micro_losses)))
            elapsed = time.perf_counter() - step_start
            record = {
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "global_step": global_step,
                "depth": depth,
                "loss": mean_loss,
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
                "step_seconds": elapsed,
                "tokens_per_second": args.batch_size * args.sequence_length * args.grad_accumulation / max(elapsed, 1e-9),
                "vram_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            }
            if global_step % args.log_every == 0 or global_step == 1:
                with log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if mean_loss < history.get("best_loss", float("inf")):
                history["best_loss"] = mean_loss
                torch.save(
                    checkpoint_payload(model, optimizer, scheduler, epoch, batch_in_epoch, global_step, permutation, permutation_generator, history, depth_rng),
                    run_dir / "best.pt",
                )
            if global_step % args.checkpoint_every == 0:
                torch.save(
                    checkpoint_payload(model, optimizer, scheduler, epoch, batch_in_epoch, global_step, permutation, permutation_generator, history, depth_rng),
                    run_dir / "last.pt",
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                epoch = args.epochs
                break
        else:
            epoch += 1
            permutation = torch.empty(0, dtype=torch.int64)
            batch_in_epoch = 0
            continue
        break

    torch.save(
        checkpoint_payload(model, optimizer, scheduler, epoch, batch_in_epoch, global_step, permutation, permutation_generator, history, depth_rng),
        run_dir / "last.pt",
    )
    (run_dir / "summary.json").write_text(json.dumps({
        "run_name": run_name,
        "model": args.model,
        "config": config.to_dict(),
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "epochs_requested": args.epochs,
        "epochs_completed": epoch,
        "global_steps": global_step,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "grad_accumulation": args.grad_accumulation,
        "learning_rate": args.lr,
        "elapsed_seconds": time.perf_counter() - start_time,
        "status": "complete" if epoch >= args.epochs else "stopped",
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
