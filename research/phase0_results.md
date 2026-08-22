# Phase 0 results

## Strongest prior art

The broad question—recurrent latent depth, adaptive computation, token/sample
selectivity, and an explicit compute/value signal—is not novel. The strongest
direct threats are ANIRA (ICML 2026) for online token-adaptive algorithmic
depth, Two-Scale for trajectory-based exits, and Looped State-Space Language
Models with Adaptive Exit-State Selection for looped SSMs plus realized
next-loss-improvement supervision. Mixture-of-Depths, Sparse Universal
Transformers, Coconut, and recurrent-depth latent reasoning close the remaining
conceptual space from other directions.

## Surviving gap

Only a narrow, falsifiable extension survives: predict the conditional
next-token cross-entropy reduction of another tied latent step from the current
state, on a fixed causal Mamba-3 context, and use that estimate to justify safe
post-stream token skipping. This is distinct from a posterior confidence/entropy
exit, a KL/convergence exit, a realized-loss hindsight label, or global
test-time sample allocation. It is not a claim of broad novelty.

## Primary hypothesis

For token `i` at latent depth `d`, a predictor of

`E[L_{d+1}(i) - L_d(i) | z_i^(d), H_{<=i}]`

will be heterogeneous across tokens and will support a lower expected compute
cost at a fixed loss tolerance than always using the maximum tested depth.
The operational depth is the smallest tested depth whose loss is within
`epsilon` of the per-token minimum over `{0,1,2,4,8}`.

## Falsification and Phase 1 decision

The claim is falsified if marginal gains are homogeneous, if no subset is
helped, if deeper steps are uniformly required, if the value predictor cannot
beat entropy/KL/convergence controls, or if skipping cannot reduce measured
wall-clock cost. Phase 0 therefore allowed Phase 1 to proceed only as a narrow
measurement study; it did not establish novelty.
