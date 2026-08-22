"""A portable PyTorch reference for the Project Alpha Phase-1 model.

The Mamba-3 implementation below follows the public SISO equations from the
official Mamba-3 implementation, but uses chunked PyTorch matrix operations.
It is deliberately kept separate from the optional Triton kernel because the
kernel must be validated independently on every target accelerator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_float = x.float()
        inv_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inv_rms * self.weight.float()).to(dtype=x.dtype)


def heavy_tail_activation(x: Tensor) -> Tensor:
    """Positive data-dependent A activation used by Mamba-3."""
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


def rotate_pairs(x: Tensor, angles: Tensor, rotary_dim: int) -> Tensor:
    """Apply Mamba-3's pairwise rotary update to the leading dimensions."""
    if rotary_dim == 0:
        return x
    rotary_dim = min(rotary_dim, x.shape[-1])
    rotary_dim -= rotary_dim % 2
    x_rot = x[..., :rotary_dim].reshape(*x.shape[:-1], rotary_dim // 2, 2)
    cos = torch.cos(angles[..., : rotary_dim // 2]).to(dtype=x_rot.dtype)
    sin = torch.sin(angles[..., : rotary_dim // 2]).to(dtype=x_rot.dtype)
    x0, x1 = x_rot.unbind(dim=-1)
    rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return torch.cat((rotated.reshape(*x.shape[:-1], rotary_dim), x[..., rotary_dim:]), dim=-1)


class Mamba3Reference(nn.Module):
    """SISO Mamba-3 recurrence expressed as differentiable PyTorch ops.

    The public Mamba-3 SISO kernel computes the same recurrence in chunks.  A
    chunked reference keeps the quadratic-looking local product bounded by
    ``chunk_size`` while retaining an exact causal state transition between
    chunks.  This is not intended to compete with a tuned Triton kernel.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        rope_fraction: float = 0.5,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if d_state <= 0 or headdim <= 0 or d_model * expand % headdim:
            raise ValueError("d_state, headdim, and d_model*expand must define integral heads")
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.chunk_size = chunk_size
        self.rope_fraction = rope_fraction
        self.num_rope_angles = max(1, int(d_state * rope_fraction) // 2)
        self.rotary_dim = min(d_state, int(2 * self.num_rope_angles))

        # [z, x, B, C, dt, A, trapezoid, angles], matching the official SISO
        # implementation.  B/C are shared across heads in SISO mode.
        d_in_proj = (
            2 * self.d_inner
            + 2 * d_state
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        dt_min, dt_max, dt_init_floor = 1e-3, 1e-1, 1e-4
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Strict lower-triangular masks prevent a token from attending to its
        # own current-state contribution; that contribution is the explicit
        # trapezoidal q*k term below.
        self.register_buffer(
            "strict_lower_mask",
            torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=-1),
            persistent=False,
        )

    def forward(self, u: Tensor) -> Tensor:
        batch, seqlen, _ = u.shape
        projected = self.in_proj(u)
        sizes = [
            self.d_inner,
            self.d_inner,
            self.d_state,
            self.d_state,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        ]
        z, x, b_raw, c_raw, dd_dt, dd_a, trap_raw, angles = torch.split(projected, sizes, dim=-1)

        # The selective recurrence is evaluated in fp32 even under BF16 AMP.
        # This is important for long products of decay factors.
        z = z.reshape(batch, seqlen, self.nheads, self.headdim).float()
        v = x.reshape(batch, seqlen, self.nheads, self.headdim).float()
        b = self.B_norm(b_raw).float().unsqueeze(2) + self.B_bias.float().unsqueeze(0).unsqueeze(0)
        c = self.C_norm(c_raw).float().unsqueeze(2) + self.C_bias.float().unsqueeze(0).unsqueeze(0)
        b = b.expand(-1, -1, self.nheads, -1)
        c = c.expand(-1, -1, self.nheads, -1)

        dt = F.softplus(dd_dt.float() + self.dt_bias.float().view(1, 1, -1))
        a = -heavy_tail_activation(dd_a.float()).clamp_min(1e-4)
        adt = (a * dt).transpose(1, 2).contiguous()  # (B, H, L)
        trap = torch.sigmoid(trap_raw.float()).transpose(1, 2).contiguous()
        dt_h = dt.transpose(1, 2).contiguous()
        gamma = dt_h * trap
        shifted_dt = F.pad(dt_h[..., 1:], (0, 1))
        shifted_trap = F.pad(trap[..., 1:], (0, 1))
        scale = gamma + shifted_dt * (1.0 - shifted_trap)

        # q/k before rotation are needed for the same-step trapezoidal term.
        q_pre = c
        k_pre = b
        q_angles = angles.float().unsqueeze(2).expand(-1, -1, self.nheads, -1)
        q = rotate_pairs(q_pre, q_angles, self.rotary_dim)
        k = rotate_pairs(k_pre, q_angles, self.rotary_dim)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        z = z.permute(0, 2, 1, 3).contiguous()
        q_pre = q_pre.permute(0, 2, 1, 3).contiguous()
        k_pre = k_pre.permute(0, 2, 1, 3).contiguous()
        qk_self = (q_pre * k_pre).sum(dim=-1) * gamma

        state = torch.zeros(
            batch,
            self.nheads,
            self.headdim,
            self.d_state,
            device=u.device,
            dtype=torch.float32,
        )
        outputs: list[Tensor] = []
        for start in range(0, seqlen, self.chunk_size):
            end = min(start + self.chunk_size, seqlen)
            n = end - start
            q_chunk = q[:, :, start:end]
            # Mamba-3 scales K before both the intra-chunk attention term and
            # the recurrent state update.  Keeping this separate from the
            # unscaled projection is important: otherwise changing the chunk
            # partition changes the result at every chunk boundary.
            k_chunk = k[:, :, start:end] * scale[:, :, start:end].unsqueeze(-1)
            v_chunk = v[:, :, start:end]
            z_chunk = z[:, :, start:end]
            da = adt[:, :, start:end]
            cumulative = torch.cumsum(da, dim=-1)
            decay = torch.exp(cumulative)

            previous = torch.einsum("bhcn,bhpn->bhcp", q_chunk, state)
            previous = previous * decay.unsqueeze(-1)

            scores = torch.einsum("bhid,bhjd->bhij", q_chunk, k_chunk)
            relative_decay = torch.exp(
                (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).clamp_max(0.0)
            )
            causal = self.strict_lower_mask[:n, :n]
            current = torch.einsum(
                "bhij,bhjd->bhid",
                scores * relative_decay * causal,
                v_chunk,
            )
            skip = self.D.float().view(1, self.nheads, 1, 1) + qk_self[:, :, start:end].unsqueeze(-1)
            output = previous + current + skip * v_chunk
            output = output * F.silu(z_chunk)
            outputs.append(output)

            future_decay = torch.exp(cumulative[..., -1:] - cumulative)
            weighted_v = v_chunk * future_decay.unsqueeze(-1)
            state = state * decay[..., -1:].unsqueeze(-1) + torch.einsum(
                "bhcp,bhcn->bhpn", weighted_v, k_chunk
            )

        y = torch.cat(outputs, dim=2).permute(0, 2, 1, 3).reshape(batch, seqlen, self.d_inner)
        return self.out_proj(y.to(dtype=projected.dtype)).to(dtype=u.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class StreamBlock(nn.Module):
    def __init__(self, config: "ModelConfig") -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mixer = Mamba3Reference(
            d_model=config.d_model,
            d_state=config.d_state,
            expand=config.mamba_expand,
            headdim=config.headdim,
            chunk_size=config.mamba_chunk_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mixer(self.norm(x))


class LatentReasoner(nn.Module):
    """One weight-tied recurrent cross-attention + SwiGLU cell."""

    def __init__(self, config: "ModelConfig") -> None:
        super().__init__()
        d = config.d_model
        if d % config.reasoner_heads:
            raise ValueError("d_model must be divisible by reasoner_heads")
        self.d_model = d
        self.heads = config.reasoner_heads
        self.head_dim = d // self.heads
        self.q_norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.post_norm = RMSNorm(d)
        self.mlp_norm = RMSNorm(d)
        self.mlp = SwiGLU(d, config.reasoner_mlp_hidden)

    def context(self, h: Tensor) -> tuple[Tensor, Tensor]:
        batch, seqlen, _ = h.shape
        k = self.k_proj(h).view(batch, seqlen, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch, seqlen, self.heads, self.head_dim).transpose(1, 2)
        return k, v

    def step(self, z: Tensor, context: tuple[Tensor, Tensor]) -> Tensor:
        k, v = context
        batch, seqlen, _ = z.shape
        q = self.q_proj(self.q_norm(z)).view(batch, seqlen, self.heads, self.head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        a = a.transpose(1, 2).reshape(batch, seqlen, self.d_model)
        u = z + self.out_proj(a)
        return u + self.mlp(self.mlp_norm(u))


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 256
    n_stream_blocks: int = 6
    d_state: int = 64
    mamba_expand: int = 2
    headdim: int = 64
    mamba_chunk_size: int = 64
    reasoner_heads: int = 4
    reasoner_mlp_hidden: int = 384
    max_depth: int = 8
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class AlphaLM(nn.Module):
    """Causal LM with a Mamba-3 stream and optional tied latent reasoner."""

    def __init__(self, config: ModelConfig, with_reasoner: bool = True) -> None:
        super().__init__()
        self.config = config
        self.with_reasoner = with_reasoner
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # A tied LM head sees RMS-normalized hidden states.  Use the usual
        # small language-model embedding scale rather than nn.Embedding's
        # unit-variance default, which would make the initial softmax far too
        # sharp (and destabilize the recurrent experiment).
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.stream = nn.ModuleList([StreamBlock(config) for _ in range(config.n_stream_blocks)])
        self.stream_norm = RMSNorm(config.d_model)
        self.reasoner = LatentReasoner(config) if with_reasoner else None

    def encode_stream(self, input_ids: Tensor) -> Tensor:
        h = self.token_embedding(input_ids)
        for block in self.stream:
            h = block(h)
        return self.stream_norm(h)

    def logits(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.token_embedding.weight)

    def hidden_at_depths(
        self,
        input_ids: Tensor,
        depths: Iterable[int],
        return_diagnostics: bool = False,
    ) -> tuple[dict[int, Tensor], dict[str, Tensor]]:
        requested = sorted(set(int(d) for d in depths))
        if any(d < 0 or d > self.config.max_depth for d in requested):
            raise ValueError(f"depths must be in [0, {self.config.max_depth}]")
        h = self.encode_stream(input_ids)
        hidden = {0: h} if 0 in requested else {}
        diagnostics: dict[str, Tensor] = {}
        if return_diagnostics:
            diagnostics["hidden_norm_0"] = h.float().norm(dim=-1)
        if not self.with_reasoner:
            if any(d != 0 for d in requested):
                raise ValueError("a baseline without a reasoner only supports depth 0")
            return hidden, diagnostics

        if max(requested, default=0) == 0:
            return hidden, diagnostics

        context = self.reasoner.context(h)
        z = h
        for depth in range(1, max(requested, default=0) + 1):
            previous = z
            z = self.reasoner.step(z, context)
            if depth in requested:
                hidden[depth] = z
            if return_diagnostics:
                diagnostics[f"hidden_norm_{depth}"] = z.float().norm(dim=-1)
                diagnostics[f"step_norm_{depth}"] = (z.float() - previous.float()).norm(dim=-1)
        return hidden, diagnostics

    def loss_at_depth(self, hidden: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
        logits = self.logits(hidden)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction=reduction).reshape(
            targets.shape
        ) if reduction == "none" else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction=reduction)

    def training_loss(self, input_ids: Tensor, targets: Tensor, depth: int) -> tuple[Tensor, dict[int, Tensor]]:
        if not self.with_reasoner:
            h = self.encode_stream(input_ids)
            loss = self.loss_at_depth(h, targets)
            return loss, {0: loss}
        if depth < 1 or depth > self.config.max_depth:
            raise ValueError("reasoner training depth must be positive and within max_depth")
        h = self.encode_stream(input_ids)
        context = self.reasoner.context(h)
        z = h
        losses: dict[int, Tensor] = {}
        for r in range(1, depth + 1):
            z = self.reasoner.step(z, context)
            losses[r] = self.loss_at_depth(z, targets)
        if depth == 1:
            return losses[1], losses
        aux_weight = 0.30
        weights = {r: aux_weight / (depth - 1) for r in range(1, depth)}
        weights[depth] = 1.0 - aux_weight
        total = sum(weights[r] * losses[r] for r in losses)
        return total, losses


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))
