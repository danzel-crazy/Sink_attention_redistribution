"""Analytic FLOPs for LLaVA-style stacks.

LLM convention, matching SparseVLMs/llava/model/language_model/modelling_sparse_llama.py:234
(itself the FastV paper's formula), per decoder layer over n tokens:

    4*n*d^2 + 2*n^2*d + 3*n*d*m

The 3*n*d*m term is LLaMA's SwiGLU (gate/up/down). Papers that assume a vanilla 2-matrix FFN
use 2*n*d*m, and the formula counts a multiply-add as 1 rather than 2, so numbers produced here
must not be compared against externally reported FLOPs. Recompute every method with this module.

Norms, softmax, rotary embeddings and lm_head are excluded: they are O(n*d) and identical across
pruning methods at equal n, so they cannot move a comparison.
"""

import torch.nn as nn


def llm_layer_flops(n: int, d: int, m: int) -> float:
    """One LLaMA decoder layer over n tokens. Attention is 4nd^2 (QKVO) + 2n^2d (scores/context),
    SwiGLU MLP is 3ndm."""
    return 4 * n * d * d + 2 * n * n * d + 3 * n * d * m


def vit_layer_flops(n: int, d: int, m: int) -> float:
    """One CLIP ViT encoder layer over n tokens. Same attention, but a 2-matrix GELU MLP."""
    return 4 * n * d * d + 2 * n * n * d + 2 * n * d * m


def llm_stack_flops(trace, d: int, m: int) -> float:
    """Sum over a per-layer token trace. `trace[i]` is the token count entering layer i, so this
    handles any pruning schedule without knowing where the method prunes."""
    return sum(llm_layer_flops(n, d, m) for n in trace)


def vision_tower_flops(vision_config) -> float:
    """CLIP ViT-L/14-336 forward. Constant across pruning methods -- they all encode the full
    image -- which is exactly why it is reported as its own column: it bounds achievable TTFT
    speedup no matter how aggressively the LLM stack is pruned."""
    d = vision_config.hidden_size
    m = vision_config.intermediate_size
    layers = vision_config.num_hidden_layers
    patches = (vision_config.image_size // vision_config.patch_size) ** 2
    n = patches + 1  # + CLS
    return layers * vit_layer_flops(n, d, m)


def projector_flops(projector, n_tokens: int) -> float:
    """Walk whatever nn.Linear stack the mm_projector happens to be (LLaVA-1.5 ships a 2-layer
    GELU MLP, but this stays correct if it changes)."""
    total = 0.0
    for mod in projector.modules():
        if isinstance(mod, nn.Linear):
            total += n_tokens * mod.in_features * mod.out_features
    return total


def kv_cache_bytes(trace, config, dtype_bytes: int = 2) -> float:
    """KV cache implied by a per-layer token trace.

    Deterministic, and the honest counterpart to peak memory: absolute peak is dominated by the
    ~14GB of fp16 weights and barely moves between methods, whereas this shows the pruning
    benefit directly.
    """
    heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    per_token = 2 * heads * head_dim * dtype_bytes  # K and V
    return sum(n * per_token for n in trace)
