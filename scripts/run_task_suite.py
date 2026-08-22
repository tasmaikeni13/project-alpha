#!/usr/bin/env python3
"""Check task-generator correctness and fixed-length stability."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from project_alpha.tasks import TASK_GENERATORS, fixed_length_pointer_task, make_suite


def main() -> None:
    rng = random.Random(1729)
    checks = []
    for name, generator in TASK_GENERATORS.items():
        if name in {"direct_lookup", "multi_query_recall", "distractor_retrieval"}:
            sample = generator(rng)
        else:
            sample = generator(rng, dependency_depth=4)
        checks.append({"task": name, "target": sample.target, "input_length": sample.input_length, "depth": sample.dependency_depth})
        if not sample.target or sample.input_length <= 0:
            raise AssertionError(f"invalid sample from {name}")

    lengths = {d: fixed_length_pointer_task(random.Random(100 + d), d).input_length for d in (1, 2, 4, 8, 12, 16, 24, 32)}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"fixed-length task leaked depth through length: {lengths}")

    suite = make_suite(seed=1729, n_per_tier=8)
    by_depth = defaultdict(list)
    for sample in suite:
        if sample.task == "fixed_length_pointer":
            by_depth[sample.dependency_depth].append(sample.input_length)
    payload = {
        "seed": 1729,
        "train_depths": [1, 2, 4, 8],
        "ood_depths": [12, 16, 24, 32],
        "generator_checks": checks,
        "fixed_length_by_depth": lengths,
        "suite_counts": {str(k): len(v) for k, v in by_depth.items()},
        "status": "pass",
    }
    out = Path("results/phase0_task_stability.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
