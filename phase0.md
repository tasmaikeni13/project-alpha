# PHASE 0 — NOVELTY AUDIT AND RESEARCH SPECIFICATION
## Project: Adaptive Marginal-Value Latent Reasoning on a Mamba-3 Backbone

You are acting as a senior ML researcher, research engineer, and adversarial reviewer.

Repository:
`tasmaikeni13/project-alpha`

The long-term project is a reasoning-first language architecture using Mamba-3 as the stream-modeling backbone and a weight-tied latent reasoning mechanism that can spend variable computation on different tokens/problems without emitting chain-of-thought.

DO NOT implement the full architecture yet.

Your job in this phase is to determine whether the proposed research question is:
1. genuinely scientifically interesting,
2. sufficiently differentiated from existing literature,
3. falsifiable,
4. experimentally feasible,
5. capable of producing useful negative results even if the proposed method fails.

---

## CORE RESEARCH IDEA

We explicitly DO NOT claim that assigning a prompt to AC0, TC0, NC1, NC2, etc. is mathematically meaningful.

Circuit-complexity classes inspired the intuition only.

Instead define an operational quantity:

For token/example i after latent reasoning step r,

    l_i^(r) = task loss after r recurrent reasoning steps

Define realized marginal compute value:

    Delta_i,r = l_i^(r) - l_i^(r+1)

Define the conditional marginal value of another reasoning iteration:

    g_i,r = E[Delta_i,r | h_i^(r), context, remaining_budget]

The candidate adaptive controller predicts:

    g_hat_i,r = f_phi(h_i^(r), diagnostics, remaining_budget)

and continues computation if roughly:

    g_hat_i,r / cost_r > lambda

subject to a trusted hard maximum compute budget.

The controller has no access to future losses at inference time.
Future losses are allowed ONLY as hindsight training targets.

The research question is:

> Can a recurrent latent reasoner learn the marginal value of additional computation and allocate depth to tokens/problems where extra computation actually improves prediction, rather than merely correlating depth with superficial notions of difficulty?

Secondary question:

> Does this allocation extrapolate to computational regimes harder than those observed during training?

No emitted chain-of-thought is required.

---

## FIRST TASK: CURRENT LITERATURE AUDIT

Search the internet extensively.

At minimum inspect and compare:

- Adaptive Computation Time — Graves
- Universal Transformer
- PonderNet
- Mixture-of-Depths
- Sparse Universal Transformer
- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach
- Coconut / continuous latent reasoning
- ANIRA: Understanding Dynamic Compute Allocation in Recurrent Transformers, ICML 2026
- Two-Scale Latent Dynamics for Recurrent-Depth Transformers
- Token-Selective Attention / learned adaptive token routing
- BUDDY / dynamic depth routing
- any 2025–2026 adaptive test-time-compute papers
- any methods predicting expected improvement, value-of-computation, marginal utility, uncertainty, convergence, residual change, or loss decrease to decide whether to continue
- Mamba/Mamba-2/Mamba-3 recurrent or looped reasoning work
- adaptive reasoning specifically on SSMs
- any work coupling Mamba/SSM representations to recurrent latent reasoning

Search papers, OpenReview, arXiv, GitHub, conference proceedings and very recent preprints.

Do not rely on title similarity.
Read methods.

Create:

`research/literature_matrix.md`

with columns:

| Work | Year | Architecture | Adaptive granularity | Decision timing | Halting target/signal | Explicit compute budget? | Latent reasoning? | Hindsight supervision? | OOD complexity tested? | Real wall-clock saved? | Overlap with us |

---

## ADVERSARIAL NOVELTY TEST

Try hard to find prior art equivalent to:

1. predicting expected NEXT-ITERATION loss reduction;
2. using that prediction as marginal benefit / compute-cost;
3. online recurrent latent reasoning;
4. token- or sample-dependent compute;
5. explicit compute-budget conditioning;
6. complexity extrapolation;
7. no textual CoT.

If a prior work already does essentially this, DO NOT hide it.

Instead:
- identify exact overlap,
- identify what remains open,
- narrow or replace our hypothesis.

Do not use the word "novel" unless evidence supports it.

---

## FORMAL DEFINITIONS

Write:

`research/hypothesis.md`

Formalize:

### H0
Extra latent recurrence has no useful heterogeneous marginal value.

### H1
Marginal value varies substantially across examples/tokens and reasoning states.

### H2
A predictor using current latent state can estimate marginal value better than trivial baselines.

### H3
A marginal-value controller gives a better accuracy/compute Pareto frontier than:
- fixed depth,
- early depth prediction,
- ACT/PonderNet-style halting,
- ANIRA-style online halting or closest reproducible equivalent,
- representation-drift/KL early exit.

### H4
The learned allocation responds to computational structure rather than only:
- sequence length,
- token frequency,
- entropy,
- position,
- input length,
- lexical features.

### H5
Allocation partially extrapolates to unseen computational depth.

---

## OPERATIONAL COMPLEXITY

Do NOT use AC0/TC0/NC1 as labels.

Define an empirical quantity:

    d*_epsilon(x)

as the minimum reasoning depth d such that:

    loss_d(x) <= min_j loss_j(x) + epsilon

or, for exact tasks, the minimum depth reaching a target correctness threshold.

Also define:

    marginal compute curve:
    M_x(d) = loss_(d-1)(x) - loss_d(x)

Call this "required latent depth" or "empirical compute demand."

If useful, define difficulty tiers L0, L1, L2, ... based on generator-controlled sequential dependency depth.

---

## DESIGN THE MINIMUM ARCHITECTURE

Do not implement full Project Alpha.

Specify a minimal prototype:

1. Mamba-3 stream encoder/backbone.
2. Frozen-context latent reasoning interface.
3. One weight-tied reasoning cell.
4. Deep-supervision output head available at every recurrence.
5. Online controller.
6. Hard trusted maximum depth.

The reasoner should be capable of inspecting encoded causal context while refining the current latent token state.

Compare possible designs:

A. repeated Mamba-3 reasoning cell;
B. recurrent cross-attention + SwiGLU cell over frozen Mamba context;
C. small recurrent Transformer cell.

Select one based on:
- scientific cleanliness,
- ability to vary compute per token/example,
- actual ability to skip computation at inference,
- GPU efficiency,
- implementation feasibility on NVIDIA and AMD ROCm.

Explain the choice.

---

## DESIGN CONTROLLED TASKS

Create task generators where true sequential dependency can be manipulated.

Include at least:

- direct associative lookup;
- multi-query associative recall;
- pointer chasing;
- adaptive pointer chasing where the next address depends on the previous retrieved value;
- state tracking;
- iterated function composition;
- distractor-controlled retrieval;
- at least one task where input length is held constant while required sequential depth changes.

Critical requirement:

Difficulty must not be inferable solely from sequence length.

Train range and OOD test range must be explicitly separated.

Example:
- train D in {1,2,4,8}
- OOD D in {12,16,24,32}

Do not finalize values until checking task stability.

---

## SYSTEMS PLAN

Target environments:

### Early development
NVIDIA RTX 3050 6 GB or whatever hardware is detected.

### Scale-up
1x AMD Instinct MI300X.

Use:
- Python 3.x
- PyTorch
- BF16 where supported
- Triton for custom kernels
- FlashAttention/SDPA optimized backend where applicable
- torch.compile only when benchmarked and stable

DO NOT write a pure-C++ training stack.

Hierarchy:

1. correct PyTorch reference;
2. built-in fused PyTorch/BLAS kernels;
3. FlashAttention / optimized existing kernels;
4. Triton;
5. C++/HIP custom op only after profiler evidence.

A HIP kernel is justified only if:
- the operation is >=10% of measured relevant wall-clock, AND
- optimized PyTorch/Triton cannot get satisfactory performance.

---

## DELIVERABLES

Produce:

- research/literature_matrix.md
- research/hypothesis.md
- research/architecture_v0.md
- research/task_suite.md
- research/novelty_risks.md
- research/phase1_plan.md

At the end print:

1. strongest prior-art threat;
2. strongest surviving research gap;
3. exact primary hypothesis;
4. exact falsification experiment;
5. go/no-go recommendation.

Do not begin Phase 1 automatically if the novelty audit destroys the research question.
Instead propose the narrowest defensible replacement.
