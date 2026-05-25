from __future__ import annotations

import torch

from qwen_refactor.config import default_config
from qwen_refactor.generation import GenerationConfig, TokenGenerator
from qwen_refactor.model import TextModel


def main() -> None:
    config = default_config()
    model = TextModel(config)
    generator = TokenGenerator(
        model,
        GenerationConfig(
            max_new_tokens=12,
            temperature=0.9,
            top_k=20,
            top_p=0.95,
            eos_token_id=config.eos_token_id,
            seed=7,
        ),
    )

    batch_size = 2
    prompt_len = 8
    prompt_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len))
    generated_ids = generator.generate(prompt_ids)

    print('prompt ids:', prompt_ids)
    print('generated ids:', generated_ids)
    print('generated shape:', generated_ids.shape)


if __name__ == '__main__':
    main()
