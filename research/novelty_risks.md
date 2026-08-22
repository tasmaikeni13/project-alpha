# Adversarial novelty and risk register

## Strongest prior-art threats

1. **Looped State-Space Language Models with Adaptive Exit-State Selection**
   ([Yu et al., 2026](https://arxiv.org/abs/2607.10110)) is the closest threat.
   It repeatedly applies a shared Mamba block, trains an exit gate, and uses
   realized improvement between successive losses as a refinement target. It
   also says that output selection does not save wall-clock time when recurrent
   SSM state continuity requires all steps. This defeats any broad claim that
   Mamba plus looped adaptive exits or hindsight improvement targets are new.

2. **ANIRA** ([Moosa et al., 2026](https://arxiv.org/abs/2602.08864)) already
   supplies unified recurrent Transformer models, online token-wise halting,
   controlled algorithmic complexity, and OOD size tests. Its reported failure
   to extrapolate is not evidence for this project; it is a mandatory baseline
   and a warning that apparent difficulty alignment can be size-specific.

3. **Marginal-reward allocation** ([Snell et al.,
   2025](https://openreview.net/forum?id=4FWAwZtd2n)) predicts the benefit of
   another unit of test-time computation and allocates a finite global budget.
   **Manvi et al.** ([2024](https://arxiv.org/abs/2410.02725)) predict whether
   another sample/restart can improve a response. Value prediction must
   therefore be distinguished by token-level CE targets over a recurrent latent
   trajectory, not by the phrase “marginal value.”

4. Recurrent latent depth and convergence exits are established by
   [Geiping et al. (2025)](https://arxiv.org/abs/2502.05171) and
   [Pappone et al. (2025)](https://arxiv.org/abs/2509.23314). KL, residual
   drift, and second-order trajectory signals must be compared directly.

5. Token routing and hard budget control are represented by
   [Mixture-of-Depths](https://arxiv.org/abs/2404.02258),
   [Token-Selective Attention](https://arxiv.org/abs/2605.05222), and
   [BUDDY](https://arxiv.org/abs/2606.09514). A claimed speedup must be an
   end-to-end synchronized speedup, not token-layer arithmetic.

## Narrow surviving gap

The defensible extension is the joint study of:

- a Mamba-3 SISO causal encoder;
- a frozen-context recurrent cross-attention latent cell;
- realized **next-token cross-entropy reduction** as a hindsight target;
- conditional prediction from the online latent state and remaining budget;
- token skipping that is semantically safe because it happens after the SSM
  stream, rather than pretending an SSM token can halt independently; and
- generator-controlled, fixed-length, unseen sequential dependency depths with
  real MI300X latency measurements.

Even this conjunction should be described as an untested combination, not
automatically as a novel method. The experiment can still produce a useful
negative result if value prediction reduces to entropy/position, allocation
fails OOD, or active-token packing does not save wall time.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Mamba-3 Triton portability failure on AMD | unit-test official path; use equation-level PyTorch reference; record exact failure; do not present fallback as a fast-kernel result |
| recurrence sharpens every prediction equally | store full token curves and report helped/harmed/zero fractions and (d^*\) histograms |
| frequency or entropy explains allocation | balanced controls, feature-only baselines, fixed-length task, held-out generator depths |
| later controller leaks future loss | train from stored hindsight labels but expose only current state/diagnostics at inference |
| output selection masquerades as savings | count executed cell work and measure packed active-token wall time with synchronization |
| corpus/tokenizer silently changes | save raw hashes, tokenizer hash, vocabulary, and exact token counts |
| negative gains disappear in averages | retain per-token arrays, examples, marginal distributions, and bootstrap intervals |
