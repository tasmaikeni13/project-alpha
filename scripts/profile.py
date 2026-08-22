#!/usr/bin/env python3
"""Synchronized component and end-to-end profiling for Phase 1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from project_alpha.data import PackedTokenDataset
from project_alpha.model import AlphaLM, ModelConfig
from evaluate import load_model


ROOT = Path(__file__).resolve().parents[1]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device: torch.device, warmup: int = 2, repeats: int = 8) -> tuple[float, object]:
    for _ in range(warmup):
        result = fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    sync(device)
    return (time.perf_counter() - start) / repeats, result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default="results/phase1/profile.json")
    p.add_argument("--torch-profiler", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = json.loads((ROOT / "data" / "processed" / "metadata.json").read_text())
    model, _ = load_model(ROOT / args.checkpoint, device)
    dataset = PackedTokenDataset(ROOT / metadata["token_paths"]["validation"], args.sequence_length)
    inputs, targets = dataset.batch(list(range(args.batch_size)), device)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        stream_time, h = timed(lambda: model.encode_stream(inputs), device)
        context_time, context = timed(lambda: model.reasoner.context(h), device)
        qkv_time, _ = timed(lambda: model.reasoner.q_proj(model.reasoner.q_norm(h)), device)
        reasoner_time, z = timed(lambda: model.reasoner.step(h, context), device)
        mlp_time, _ = timed(lambda: model.reasoner.mlp(model.reasoner.mlp_norm(h)), device)
        output_time, _ = timed(lambda: model.logits(h), device)
        recurrence_loop_time, _ = timed(lambda: [None for _ in range(8)], device, warmup=2, repeats=1000)
        full_times = {}
        for depth in (0, 1, 2, 4, 8):
            elapsed, _ = timed(lambda d=depth: model.hidden_at_depths(inputs, [d])[0][d], device)
            full_times[str(depth)] = {"seconds": elapsed, "tokens_per_second": args.batch_size * args.sequence_length / elapsed}

    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "tokens": args.batch_size * args.sequence_length,
        "component_seconds": {
            "mamba_stream": stream_time,
            "context_kv_projection": context_time,
            "reasoning_q_projection": qkv_time,
            "reasoning_attention_plus_mlp": reasoner_time,
            "reasoning_mlp_only": mlp_time,
            "output_projection": output_time,
            "python_recurrence_loop_1000x": recurrence_loop_time,
        },
        "full_forward": full_times,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "profiler_trace": None,
    }
    if args.torch_profiler:
        trace_dir = ROOT / "results" / "phase1" / "torch_profiler"
        trace_dir.mkdir(parents=True, exist_ok=True)
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
            ) as prof:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    model.hidden_at_depths(inputs, [4])
                sync(device)
            prof.export_chrome_trace(str(trace_dir / "phase1_forward.json"))
            result["profiler_trace"] = "results/phase1/torch_profiler/phase1_forward.json"
        except Exception as exc:
            result["profiler_error"] = repr(exc)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
