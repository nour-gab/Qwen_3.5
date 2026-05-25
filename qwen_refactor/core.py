from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def build(self, position_ids: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(reference.shape[0], -1)

        inv_freq = self.inv_freq.to(device=reference.device, dtype=reference.dtype)
        angles = position_ids.float().unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos().to(dtype=reference.dtype), angles.sin().to(dtype=reference.dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    def apply(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rotary_dim = cos.shape[-1]
        q_rot = q.clone()
        k_rot = k.clone()
        q_main = q_rot[..., :rotary_dim]
        k_main = k_rot[..., :rotary_dim]
        q_rot[..., :rotary_dim] = q_main * cos - self.rotate_half(q_main) * sin
        k_rot[..., :rotary_dim] = k_main * cos - self.rotate_half(k_main) * sin
        return q_rot, k_rot


def build_causal_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    kv_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    causal = torch.full((query_length, kv_length), min_value, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1 + kv_length - query_length)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_length, kv_length)

    if attention_mask is None:
        return causal

    mask = attention_mask.to(device=device)
    if mask.dtype != torch.bool:
        mask = mask != 0
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    else:
        while mask.dim() < 4:
            mask = mask.unsqueeze(1)

    mask = mask.expand(batch_size, 1, query_length, kv_length)
    padding = (~mask).to(dtype) * min_value
    return causal + padding
