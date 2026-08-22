#!/usr/bin/env python3
"""Record the exact Phase-1 software and accelerator environment."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(cmd: list[str]) -> dict:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
        return {"command": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"command": cmd, "error": repr(exc)}


def main() -> None:
    import torch
    import triton

    gpu = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "gcn_arch": getattr(props, "gcnArchName", None),
            "total_memory_bytes": props.total_memory,
            "major": props.major,
            "minor": props.minor,
            "multiprocessors": props.multi_processor_count,
        }
    payload = {
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "gpu": gpu,
        "accelerator": {"cuda_available": torch.cuda.is_available(), "torch_cuda_api": torch.version.cuda, "torch_hip": torch.version.hip},
        "software": {"torch": torch.__version__, "triton": triton.__version__},
        "compiler_toolchain": {
            "gcc": command(["gcc", "--version"]),
            "hipcc": command(["hipcc", "--version"]),
            "hipconfig": command(["hipconfig", "--full"]),
            "rocm_smi": command(["rocm-smi", "--showproductname", "--showmeminfo", "vram"]),
        },
        "commands": {
            "rocminfo": command(["rocminfo"]),
            "rocm_smi_full": command(["rocm-smi"]),
        },
        "official_mamba3": {
            "source": "https://github.com/state-spaces/mamba",
            "commit": "e9594ce1c732d97440f0332fdc43170a2294dbfa",
            "status": "Triton SISO kernel import succeeded; MI300X forward probe failed during LLVM register allocation; Phase-1 uses the PyTorch equation-level fallback.",
        },
        "triton_kernel_policy": "No custom HIP/CUDA kernel; no unverified Mamba-3 Triton kernel in the training result.",
    }
    out = ROOT / "runs" / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
