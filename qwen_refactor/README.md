# Qwen Refactor Package

This package rewrites the model into smaller object-oriented modules and adds an explicit generation layer.

## Layout
- [docs/overview.md](docs/overview.md) is the entry point for the architecture diagrams.
- [docs/generation.md](docs/generation.md) explains the autoregressive sampler.
- [docs/model.md](docs/model.md) explains the forward pass.
- [docs/decoder.md](docs/decoder.md) explains block composition.
- [docs/core.md](docs/core.md), [docs/cache.md](docs/cache.md), [docs/attention.md](docs/attention.md), [docs/gated_delta.md](docs/gated_delta.md), and [docs/mlp.md](docs/mlp.md) cover the lower-level pieces.

## Run
```bash
python -m qwen_refactor.main
```

## What the demo does
The runner now generates tokens end to end from a random prompt, using the model cache and sampling policy.
