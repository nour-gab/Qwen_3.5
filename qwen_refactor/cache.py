from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ModelCache:
    num_layers: int
    keys: list[Optional[torch.Tensor]] = field(init=False)
    values: list[Optional[torch.Tensor]] = field(init=False)
    recurrent_states: list[Optional[torch.Tensor]] = field(init=False)
    conv_states: list[Optional[torch.Tensor]] = field(init=False)

    def __post_init__(self) -> None:
        self.keys = [None for _ in range(self.num_layers)]
        self.values = [None for _ in range(self.num_layers)]
        self.recurrent_states = [None for _ in range(self.num_layers)]
        self.conv_states = [None for _ in range(self.num_layers)]

    def update_kv(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = keys
            self.values[layer_idx] = values
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], keys], dim=-2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], values], dim=-2)
        return self.keys[layer_idx], self.values[layer_idx]

    def get_seq_len(self) -> int:
        for keys in self.keys:
            if keys is not None:
                return keys.shape[-2]
        return 0
