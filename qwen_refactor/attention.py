from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from qwen_refactor.cache import ModelCache
from qwen_refactor.core import RMSNorm, RotaryEmbedding, build_causal_mask
from qwen_refactor.config import ModelConfig


class FullAttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(self.head_dim, config.theta)

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: ModelCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q_proj = self.q_proj(hidden_states)
        query, gate = torch.chunk(q_proj, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)

        query = self.q_norm(query.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary.build(pos_ids, hidden_states)
        query, key = self.rotary.apply(query, key, cos, sin)

        if cache is not None:
            key, value = cache.update_kv(self.layer_idx, key, value)

        key = key.repeat_interleave(self.num_kv_groups, dim=1)
        value = value.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            mask = attention_mask
            if mask.dtype != torch.bool:
                mask = mask != 0
            while mask.dim() < attn_weights.dim():
                mask = mask.unsqueeze(1)
            mask = mask.expand_as(attn_weights)
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.out_proj(attn_output)
