from __future__ import annotations

from torch import nn
import torch

from qwen_refactor.attention import FullAttentionBlock
from qwen_refactor.config import ModelConfig
from qwen_refactor.core import RMSNorm
from qwen_refactor.gated_delta import GatedDeltaNet
from qwen_refactor.mlp import GatedMLP


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.input_layer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)
        self.attention = self._build_attention(config)

    def _build_attention(self, config: ModelConfig) -> nn.Module:
        if self.layer_type == "Linear_attention":
            return GatedDeltaNet(config, self.layer_idx)
        return FullAttentionBlock(config, self.layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layer_norm(hidden_states)

        if self.layer_type == "Linear_attention":
            attn_output = self.attention(hidden_states, attention_mask=attention_mask, cache=cache)
        else:
            attn_output = self.attention(hidden_states, pos_ids=pos_ids, attention_mask=attention_mask, cache=cache)

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states
