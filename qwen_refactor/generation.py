from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from qwen_refactor.cache import ModelCache
from qwen_refactor.model import TextModel


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: float = 1.0
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    seed: Optional[int] = 0


class TokenSampler:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.temperature <= 0:
            return torch.argmax(logits, dim=-1)

        scores = torch.nan_to_num(logits / self.config.temperature, nan=-1e9, posinf=-1e9, neginf=-1e9)
        if self.config.top_k is not None and self.config.top_k > 0:
            top_k = min(self.config.top_k, scores.shape[-1])
            top_values, top_indices = torch.topk(scores, top_k, dim=-1)
            filtered = torch.full_like(scores, float("-inf"))
            filtered.scatter_(-1, top_indices, top_values)
            scores = filtered

        if self.config.top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > self.config.top_p
            sorted_mask[..., 0] = False
            sorted_scores = sorted_scores.masked_fill(sorted_mask, float("-inf"))
            scores = torch.full_like(scores, float("-inf"))
            scores.scatter_(-1, sorted_indices, sorted_scores)

        probs = torch.softmax(scores, dim=-1)
        if not torch.isfinite(probs).all() or torch.sum(probs, dim=-1).eq(0).any():
            return torch.argmax(scores, dim=-1)

        if self.config.do_sample:
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        return torch.argmax(probs, dim=-1)


class TokenGenerator:
    def __init__(self, model: TextModel, config: GenerationConfig | None = None) -> None:
        self.model = model
        self.config = config or GenerationConfig()
        self.sampler = TokenSampler(self.config)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        generated = input_ids
        cache: ModelCache | None = None

        for _ in range(self.config.max_new_tokens):
            logits, cache = self.model(generated[:, -1:] if cache is not None else generated, cache=cache)
            next_token = self.sampler.sample(logits[:, -1, :]).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)

            if self.config.eos_token_id is not None:
                if torch.all(next_token.squeeze(-1) == self.config.eos_token_id):
                    break

        return generated
