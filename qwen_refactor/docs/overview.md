# Overview

This overview shows how the refactored package fits together from prompt to token generation.

## Forward pass sequence
```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant M as TokenGenerator
    participant T as TextModel
    participant C as ModelCache
    participant D as DecoderLayer
    participant A as Attention / Delta block
    participant F as MLP

    U->>M: prompt token ids
    M->>T: generate(input_ids)
    T->>C: create or reuse cache
    loop for each decoder layer
        T->>D: hidden states + mask + pos ids
        D->>A: normalize and attend / recurrent update
        A-->>D: attention output
        D->>F: residual + post-attn MLP
        F-->>T: next hidden states
    end
    T-->>M: logits
    M->>M: sample next token
    M-->>U: generated token ids
```

## Cache flow sequence
```mermaid
sequenceDiagram
    autonumber
    participant T as TextModel
    participant C as ModelCache
    participant F as FullAttentionBlock
    participant G as GatedDeltaNet

    T->>C: get_seq_len()
    T->>T: build mask and position ids from prefix length
    T->>F: update KV cache for full-attention layers
    F->>C: store updated keys and values
    T->>G: run recurrent delta step for linear-attention layers
    G->>C: store recurrent and conv state
    C-->>T: cached state reused for the next step
```

## Cross-links
- [Configuration](config.md)
- [Core math](core.md)
- [Cache](cache.md)
- [Full attention](attention.md)
- [Gated delta block](gated_delta.md)
- [Feed-forward block](mlp.md)
- [Decoder composition](decoder.md)
- [Model orchestration](model.md)
- [Generation](generation.md)
- [Runnable demo](main.md)
