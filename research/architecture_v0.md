# Minimal architecture v0

## Selected design

The prototype is design **B: recurrent cross-attention + SwiGLU over frozen
Mamba-3 context**.

```text
token ids -> tied embedding -> 6 x pre-norm Mamba-3 SISO stream -> H
                                      |                         |
                                      | fixed K,V projections   |
                                      v                         v
                         z0 = H -> [causal cross-attn + SwiGLU]^T -> logits
```

The one recurrent cell is weight-tied across all (T) applications:

\[
q^{(r)}=W_q\operatorname{RMSNorm}(z^{(r)}),\quad
a^{(r)}=\operatorname{SDPA}(q^{(r)},K(H),V(H),\text{causal}),
\]
\[
u^{(r)}=z^{(r)}+W_o a^{(r)},\qquad
z^{(r+1)}=u^{(r)}+\operatorname{SwiGLU}(\operatorname{RMSNorm}(u^{(r)})).
\]

The output head is tied to the input embedding and shared at every depth. There
is no adaptive halting head in Phase 1. A later controller will observe only
the current state, cheap diagnostics, and remaining budget, with a trusted
hard maximum.

The implementation is in [`src/project_alpha/model.py`](../src/project_alpha/model.py).
`Mamba3Reference` follows the public Mamba-3 SISO trapezoidal/rotary equations
with chunked PyTorch operations. The official Triton implementation from
[`state-spaces/mamba`](https://github.com/state-spaces/mamba) was imported and
probed on `gfx942`; its forward compilation failed with an LLVM register
allocation error, so it is not silently used for training. This is recorded in
[`runs/environment.json`](../runs/environment.json).

## Why design B

| Candidate | Scientific cleanliness | Variable compute | Actual skip potential | AMD/NVIDIA feasibility | Decision |
|---|---|---|---|---|---|
| A. Repeated Mamba-3 reasoner | strong SSM continuity, but recurrent SSM state couples positions | sequence depth is easy; token skipping is hard | weak without state surgery | reference feasible; fast path target-sensitive | future SSM comparison |
| **B. Frozen-context cross-attention + SwiGLU** | clean separation of one causal encoding and repeated refinement | active token states can be grouped by depth | strongest: inactive tokens can be removed while H stays fixed | SDPA/matmul are portable; no custom op initially | **selected** |
| C. Small recurrent Transformer cell | simple and portable | same as B | good | SDPA is portable | useful ablation, but less direct alignment with the frozen-context interface |

Design B does not claim that an SSM token can halt independently without state
consequences. The Mamba stream runs causally once; only post-encoding latent
refinement is conditionally executed. That distinction is required for a real
wall-clock claim.

## Parameter and numerical budget

| field | value |
|---|---:|
| vocabulary | 32,000 ByteLevel-BPE entries |
| model width | 256 |
| stream blocks | 6 Mamba-3 SISO reference blocks |
| SSM state / head dimension | 64 / 64 |
| reasoner heads / MLP hidden | 4 / 384 |
| hard evaluation depth | 8 |
| dropout | 0 |
| precision | BF16 autocast; FP32 parameters and optimizer state |

The measured reasoner count is printed by training and stored in the run
summary; it is expected to be 11,375,968 parameters with a 32k vocabulary.
The plain baseline uses seven stream blocks and is measured separately to match
the same parameter budget within the fixed 10–15M target.

The SSM scan is evaluated in FP32 inside the BF16 model to protect products of
decay factors. The reference has no custom C++/HIP/CUDA code. A custom kernel
is permitted only after synchronized profiling shows a relevant operation is
at least 10% of wall time and a PyTorch/Triton alternative is inadequate.
