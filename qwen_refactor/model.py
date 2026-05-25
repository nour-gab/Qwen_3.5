from __future__ import annotations

import torch
from torch import nn

from qwen_refactor.cache import ModelCache
from qwen_refactor.config import ModelConfig
from qwen_refactor.core import RMSNorm, build_causal_mask
from qwen_refactor.decoder import DecoderLayer


class TextModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, idx) for idx in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.output_proj.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: ModelCache | None = None,
    ) -> tuple[torch.Tensor, ModelCache]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if cache is None:
            cache = ModelCache(self.config.num_layers)

        past_seen_tokens = cache.get_seq_len()
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                seq_len + past_seen_tokens,
                device=hidden_states.device,
                dtype=torch.bool,
            )
        elif attention_mask.shape[-1] == seq_len and past_seen_tokens > 0:
            prefix = torch.ones(batch_size, past_seen_tokens, device=hidden_states.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix, attention_mask], dim=-1)

        kv_length = seq_len + past_seen_tokens
        causal_mask = build_causal_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_length=seq_len,
            kv_length=kv_length,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        pos_ids = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=hidden_states.device).expand(batch_size, -1)

        for layer in self.layers:
            layer_mask = causal_mask if layer.layer_type == "full_attention" else attention_mask[:, -seq_len:]
            hidden_states = layer(hidden_states, pos_ids=pos_ids, attention_mask=layer_mask, cache=cache)

        hidden_states = self.final_norm(hidden_states)
        logits = self.output_proj(hidden_states)
        return logits, cache
