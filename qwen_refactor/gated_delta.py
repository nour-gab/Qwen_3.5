from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from qwen_refactor.cache import ModelCache
from qwen_refactor.config import ModelConfig
from qwen_refactor.core import RMSNorm


class RecurrentDeltaRule(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape[2] != v.shape[2]:
            if v.shape[2] % q.shape[2] == 0:
                repeat_factor = v.shape[2] // q.shape[2]
                q = q.repeat_interleave(repeat_factor, dim=2)
                k = k.repeat_interleave(repeat_factor, dim=2)
            elif q.shape[2] % v.shape[2] == 0:
                repeat_factor = q.shape[2] // v.shape[2]
                v = v.repeat_interleave(repeat_factor, dim=2)
                beta = beta.repeat_interleave(repeat_factor, dim=2)
                g = g.repeat_interleave(repeat_factor, dim=2)
            else:
                raise ValueError(f"Incompatible head counts: q/k={q.shape[2]} and v={v.shape[2]}")

        batch_size, seq_len, num_heads, head_dim = v.shape
        if recurrent_state is None:
            recurrent_state = torch.zeros(batch_size, num_heads, head_dim, device=v.device, dtype=v.dtype)

        outputs = []
        for step in range(seq_len):
            q_step = q[:, step]
            k_step = k[:, step]
            v_step = v[:, step]
            g_step = torch.sigmoid(g[:, step]).unsqueeze(-1)
            beta_step = beta[:, step].unsqueeze(-1)
            recurrent_state = recurrent_state + beta_step * (v_step + torch.tanh(q_step * k_step))
            outputs.append(g_step * recurrent_state)

        return torch.stack(outputs, dim=1), recurrent_state


class GatedDeltaNet(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.kernel_size = config.linear_conv_kernel_size

        self.k_dim = self.num_k_heads * self.head_k_dim
        self.v_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.k_dim * 2 + self.v_dim

        self.norm = RMSNorm(self.hidden_size, config.rms_norm_eps)
        self.qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=config.attention_bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=config.attention_bias,
        )
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()
        self.v_proj = nn.Identity()
        self.delta_rule = RecurrentDeltaRule()
        self.z = nn.Linear(self.hidden_size, self.v_dim, bias=config.attention_bias)
        self.b = nn.Linear(self.hidden_size, self.num_v_heads, bias=config.attention_bias)
        self.a = nn.Linear(self.hidden_size, self.num_v_heads, bias=config.attention_bias)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(1e-3, 16.0)))
        self.out_proj = nn.Linear(self.v_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: ModelCache | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)

        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv(self.norm(hidden_states)).transpose(1, 2)

        recurrent_state = cache.recurrent_states[self.layer_idx] if cache is not None else None
        conv_state = cache.conv_states[self.layer_idx] if cache is not None else None
        if cache is not None:
            conv_pad = max(self.kernel_size - qkv.shape[-1], 0)
            qkv = F.pad(qkv, (conv_pad, 0), value=0.0)
            cache.conv_states[self.layer_idx] = conv_state

        conv_out = F.silu(self.conv1d(qkv)[..., -seq_len:]).transpose(1, 2)
        q, k, v = torch.split(conv_out, [self.k_dim, self.k_dim, self.v_dim], dim=-1)

        q = q.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        z = self.z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        beta = torch.sigmoid(self.b(hidden_states))
        gates = -torch.exp(self.A_log) * F.softplus(self.a(hidden_states) + self.dt_bias)

        if self.num_v_heads % self.num_k_heads == 0 and self.num_v_heads != self.num_k_heads:
            repeat_factor = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(repeat_factor, dim=2)
            k = k.repeat_interleave(repeat_factor, dim=2)

        attn_core, recurrent_state = self.delta_rule(q, k, v, gates, beta, recurrent_state)
        if cache is not None:
            cache.recurrent_states[self.layer_idx] = recurrent_state

        attn_core = attn_core.reshape(-1, self.v_dim) * F.silu(z.reshape(-1, self.v_dim))
        attn_core = attn_core.reshape(batch_size, seq_len, self.v_dim)
        return self.out_proj(attn_core)
