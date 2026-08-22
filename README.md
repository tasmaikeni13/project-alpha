# Project Alpha: Theoretical Foundations, Reference Implementation & Verification

[![Status](https://img.shields.io/badge/Status-Theory%20%26%20Formal%20Verification-blue.svg)](#)
[![Lean 4](https://img.shields.io/badge/Formal%20Proofs-Lean%204-6f42c1.svg?logo=lean)](https://leanprover.github.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> [!NOTE]
> **Project Status: Phase 0 complete; Phase 1 halted partial**
> The repository contains the theoretical foundation, Phase 0 literature/design
> audit, a portable PyTorch Mamba-3 reference implementation, reproducible data
> tooling, and partial MI300X Phase 1 artifacts. The corrected Phase 1 training
> run was intentionally stopped before final evaluation; see
> [`research/phase1_results.md`](research/phase1_results.md).

---

## 📖 Overview

**Project Alpha** introduces a next-generation neural architecture framework designed to fundamentally resolve core scaling limitations in contemporary deep sequence and vision models:

1. **Information Preservation & Invariant Measures:** Formulating layer-wise state transitions through geometrically invariant mappings to eliminate representation collapse across extreme network depths.
2. **Dynamical Stability:** Bounding gradient dispersion and phase-space distortion across long sequence horizons.
3. **Formal Verification:** Establishing mathematically rigorous theorems verified through interactive theorem proving in **Lean 4 / Mathlib**.

For the detailed mathematical specification, consult [**`alpha-architecture.md`**](alpha-architecture.md).

---

## 🏛️ Formal Mathematical Verification (Lean 4)

All foundational properties and theorems of the Alpha Architecture are formally specified and machine-checked using **Lean 4**:

- **[`lean/AlphaArchitecture.lean`](lean/AlphaArchitecture.lean):** Primary module entrypoint defining core structural types and operators.
- **[`lean/AlphaArchitecture/Theorems.lean`](lean/AlphaArchitecture/Theorems.lean):** Machine-checked proofs establishing algebraic invariance, stability bounds, and state transition properties.

### Building & Verifying the Proofs Locally:

Ensure you have [**Elan / Lean 4**](https://github.com/leanprover/elan) installed:

```bash
# Navigate to the lean directory
cd lean

# Build and verify all formal proofs
lake build
```

---

## 🗺️ Roadmap & Subsequent Milestones

- [x] **Phase 1: Mathematical Formulation** — Formal derivation of the Alpha architectural equations and invariant maps.
- [x] **Phase 2: Lean 4 Formal Verification** — Machine-checked formal proofs of core mathematical theorems (`AlphaArchitecture/Theorems.lean`).
- [x] **Phase 3: Prototype Reference Implementation** — PyTorch reference modules and autograd verifications.
- [ ] **Phase 4: High-Performance GPU Kernel Engineering** — Native HIP and Triton vectorized operators for hardware acceleration. The official Mamba-3 Triton SISO path was probed on MI300X but failed LLVM register allocation; no unverified custom kernel is used.
- [ ] **Phase 5: Large-Scale Empirical Benchmarking** — Multi-seed scaling runs across canonical vision and language benchmarks.

## Empirical artifacts

- [`research/literature_matrix.md`](research/literature_matrix.md) — adversarial prior-art audit.
- [`research/architecture_v0.md`](research/architecture_v0.md) — Mamba-3 stream plus tied latent reasoner design.
- [`research/phase0_results.md`](research/phase0_results.md) — surviving gap and falsification criteria.
- [`runs/environment.json`](runs/environment.json) — MI300X/ROCm/PyTorch and official-kernel probe record.
- [`tests/test_reference.py`](tests/test_reference.py) — causality, chunk invariance, finite backward, and parameter-budget checks.

---

## 📂 Repository Structure

```
project-alpha/
├── README.md                      # Project Overview & Theoretical Scope
├── alpha-architecture.md         # Full Theoretical Architecture Specification
├── .gitignore                    # Build Artifact & Cache Exclusion Rules
└── lean/                         # Formal Mathematical Proofs (Lean 4)
    ├── AlphaArchitecture.lean    # Module Root
    ├── AlphaArchitecture/
    │   └── Theorems.lean         # Verified Theorems & Invariance Properties
    ├── lakefile.toml             # Lake Package Configuration
    ├── lake-manifest.json        # Mathlib Dependency Manifest
    └── lean-toolchain            # Lean 4 Version Specification
```
