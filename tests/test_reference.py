"""Small CPU checks for the portable Phase-1 reference implementation."""

from __future__ import annotations

import torch

from project_alpha.model import AlphaLM, Mamba3Reference, ModelConfig, count_parameters


def test_mamba_is_causal() -> None:
    torch.manual_seed(7)
    mixer = Mamba3Reference(d_model=16, d_state=8, expand=2, headdim=8, chunk_size=4).eval()
    inputs = torch.randn(2, 13, 16)
    changed = inputs.clone()
    changed[:, -1] += 10.0
    with torch.inference_mode():
        original = mixer(inputs)
        future_changed = mixer(changed)
    torch.testing.assert_close(original[:, :-1], future_changed[:, :-1], rtol=1e-4, atol=1e-5)


def test_mamba_chunk_partition_is_stable() -> None:
    torch.manual_seed(11)
    reference = Mamba3Reference(d_model=16, d_state=8, expand=2, headdim=8, chunk_size=4).eval()
    repartitioned = Mamba3Reference(d_model=16, d_state=8, expand=2, headdim=8, chunk_size=7).eval()
    repartitioned.load_state_dict(reference.state_dict())
    inputs = torch.randn(2, 13, 16)
    with torch.inference_mode():
        left = reference(inputs)
        right = repartitioned(inputs)
    torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-5)


def test_small_reasoner_loss_is_finite() -> None:
    torch.manual_seed(13)
    config = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_stream_blocks=2,
        d_state=8,
        mamba_expand=2,
        headdim=8,
        mamba_chunk_size=4,
        reasoner_heads=4,
        reasoner_mlp_hidden=48,
        max_depth=4,
    )
    model = AlphaLM(config, with_reasoner=True)
    inputs = torch.randint(0, config.vocab_size, (2, 12))
    targets = torch.randint(0, config.vocab_size, (2, 12))
    loss, losses = model.training_loss(inputs, targets, depth=4)
    assert torch.isfinite(loss)
    assert set(losses) == {1, 2, 3, 4}
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_phase1_parameter_budget() -> None:
    config = ModelConfig()
    model = AlphaLM(config, with_reasoner=True)
    parameters = count_parameters(model)
    assert 10_000_000 <= parameters <= 15_000_000


if __name__ == "__main__":
    for test in (
        test_mamba_is_causal,
        test_mamba_chunk_partition_is_stable,
        test_small_reasoner_loss_is_finite,
        test_phase1_parameter_budget,
    ):
        test()
        print(f"PASS {test.__name__}")
