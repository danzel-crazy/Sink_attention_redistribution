"""Find the pieces we instrument, across every fork in this repo.

The forks disagree on layout: the LLaVA-repo ones (PyramidDrop, SparseVLMs) expose
`get_vision_tower()` / `get_model().mm_projector`, while FastV's HF path (llava-hf, transformers
4.39) uses `vision_tower` / `multi_modal_projector`. Everything here probes rather than assumes,
so the recorder never needs to know which fork it is running inside.
"""

import torch.nn as nn


def find_decoder_layers(model) -> nn.ModuleList:
    """The LLM decoder stack.

    Cannot just take the longest ModuleList: a CLIP encoder layer also has `self_attn` and `mlp`,
    so the vision tower's 24 layers look identical by that test. LlamaDecoderLayer is identified
    by `input_layernorm`, which CLIPEncoderLayer (layer_norm1/layer_norm2) does not have.
    """
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            first = mod[0]
            if hasattr(first, "self_attn") and hasattr(first, "input_layernorm"):
                return mod
    raise RuntimeError(
        "efficiency: could not locate the decoder stack; expected an nn.ModuleList of layers "
        "with .self_attn and .input_layernorm"
    )


def find_vision_tower(model):
    if hasattr(model, "get_vision_tower"):
        try:
            vt = model.get_vision_tower()
            if vt is not None:
                return vt
        except Exception:
            pass
    for attr in ("vision_tower", "visual"):
        vt = getattr(model, attr, None)
        if isinstance(vt, nn.Module):
            return vt
    inner = getattr(model, "model", None)
    if inner is not None:
        for attr in ("vision_tower", "visual"):
            vt = getattr(inner, attr, None)
            if isinstance(vt, nn.Module):
                return vt
    return None


def find_projector(model):
    if hasattr(model, "get_model"):
        try:
            proj = getattr(model.get_model(), "mm_projector", None)
            if proj is not None:
                return proj
        except Exception:
            pass
    for attr in ("multi_modal_projector", "mm_projector"):
        proj = getattr(model, attr, None)
        if isinstance(proj, nn.Module):
            return proj
    inner = getattr(model, "model", None)
    if inner is not None:
        for attr in ("multi_modal_projector", "mm_projector"):
            proj = getattr(inner, attr, None)
            if isinstance(proj, nn.Module):
                return proj
    return None


def find_lm_head(model):
    """Marks the end of prefill compute, so TTFT includes logits rather than stopping at the last
    decoder layer."""
    if hasattr(model, "get_output_embeddings"):
        try:
            head = model.get_output_embeddings()
            if head is not None:
                return head
        except Exception:
            pass
    return getattr(model, "lm_head", None)


def vision_config_of(vision_tower):
    """CLIPVisionTower (LLaVA repo) wraps the real tower and holds `.config`; the HF path exposes
    `.config` directly."""
    cfg = getattr(vision_tower, "config", None)
    if cfg is not None and hasattr(cfg, "hidden_size") and hasattr(cfg, "patch_size"):
        return cfg
    inner = getattr(vision_tower, "vision_model", None)
    if inner is not None:
        cfg = getattr(inner, "config", None)
        if cfg is not None:
            return cfg
    return None


def text_config_of(model):
    """Hidden/intermediate size of the LLM, not the ViT. On HF llava the top-level config nests the
    LLM under `.text_config`."""
    cfg = model.config
    if hasattr(cfg, "text_config"):
        return cfg.text_config
    return cfg
