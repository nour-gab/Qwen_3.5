from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 256
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False

    hidden_size: int = 128
    num_hidden_layers: int = 4
    layer_types: Tuple[str, ...] = (
        "Linear_attention",
        "full_attention",
        "Linear_attention",
        "full_attention",
    )

    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    attention_dropout: float = 0.0
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6

    hidden_act: str = "silu"
    intermediate_size: int = 512

    num_v_heads: int = 4
    num_k_heads: int = 2
    head_k_dim: int = 32
    head_v_dim: int = 32
    linear_conv_kernel_size: int = 4

    rotary_factor: float = 1.0
    theta: float = 100000.0
    dim: int = 32
    mrope_section: Tuple[int, ...] = (11, 11, 10)

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)


def default_config() -> ModelConfig:
    return ModelConfig()
