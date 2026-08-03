"""Method-agnostic efficiency recorder.

Works on FastV, PyramidDrop, SparseVLM, their `_cross` variants and vanilla LLaVA through one
code path, because all of them *physically truncate* `hidden_states` at their pruning layers
rather than masking. A forward-pre-hook on each decoder layer therefore observes the real token
count entering that layer, whatever the method's schedule is.

Design rules, in priority order:

1. Never perturb generation. Hooks are read-only, return None, and every hook body is wrapped so
   an instrumentation bug degrades to "no data" instead of a failed eval run. Disabled by default:
   `from_env` returns a no-op unless EFFICIENCY_MODE is set.
2. Never perturb the thing being measured. `count` mode issues no syncs at all. `bench` mode
   records CUDA events at mark points and resolves them with a single synchronize per sample,
   rather than blocking at each mark.
"""

import os
import time
import uuid
from contextlib import contextmanager

import torch

from . import flops as F
from .locate import (
    find_decoder_layers,
    find_lm_head,
    find_projector,
    find_vision_tower,
    text_config_of,
    vision_config_of,
)
from .writer import RunWriter

MODES = ("off", "count", "bench")


def _register_pre_hook(module, fn):
    """`with_kwargs` needs torch>=2.0; fall back to positional-only, which is what every fork's
    decoder loop actually uses (`decoder_layer(hidden_states, attention_mask=...)`)."""
    try:
        return module.register_forward_pre_hook(fn, with_kwargs=True)
    except TypeError:
        return module.register_forward_pre_hook(lambda m, a: fn(m, a, {}))


class _NullRecorder:
    """What every entry point gets unless EFFICIENCY_MODE is set. Same surface, does nothing."""

    enabled = False

    def attach(self):
        return self

    def detach(self):
        pass

    def should_stop(self):
        return False

    @contextmanager
    def sample(self, question_id=None):
        yield

    def close(self):
        pass


class EfficiencyRecorder:
    def __init__(self, model, method, config=None, mode="count", out_path=None,
                 run_id=None, warmup=3, limit=None, dataset=None, notes=None):
        if mode not in MODES:
            raise ValueError(f"efficiency: mode must be one of {MODES}, got {mode!r}")
        self.model = model
        self.method = method
        self.config = config or {}
        self.mode = mode
        self.bench = mode == "bench"
        self.enabled = mode != "off"
        self.warmup = warmup if self.bench else 0
        self.limit = limit
        self.dataset = dataset
        self.notes = notes
        self.run_id = run_id or f"{method}-{uuid.uuid4().hex[:8]}"
        self._writer = RunWriter(out_path, self.run_id) if self.enabled else None

        self._handles = []
        self._layers = None
        self._n_layers = 0
        self._d = None
        self._m = None
        self._vision_tflops = 0.0
        self._weights_bytes = 0

        # per-sample state
        self._sample_idx = 0
        self._written = 0
        self._trace = []
        self._forward_idx = 0
        self._decode_forwards = 0
        self._events = {}
        self._wall_start = 0.0
        self._mem_baseline = 0

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_env(cls, model, method, config=None, dataset=None):
        """The only constructor the eval loaders call. Returns a no-op recorder unless
        EFFICIENCY_MODE is set, so an un-instrumented run is byte-for-byte the run we had before.
        """
        mode = os.environ.get("EFFICIENCY_MODE", "off").lower()
        if mode == "off":
            return _NullRecorder()
        limit = os.environ.get("EFFICIENCY_LIMIT")
        rec = cls(
            model,
            method=os.environ.get("EFFICIENCY_METHOD", method),
            config=config,
            mode=mode,
            out_path=os.environ.get("EFFICIENCY_OUT"),
            run_id=os.environ.get("EFFICIENCY_RUN_ID"),
            warmup=int(os.environ.get("EFFICIENCY_WARMUP", "3")),
            limit=int(limit) if limit else None,
            dataset=os.environ.get("EFFICIENCY_DATASET", dataset),
            notes=os.environ.get("EFFICIENCY_NOTES"),
        )
        return rec.attach()

    def attach(self):
        try:
            self._attach()
        except Exception as exc:  # never take an eval run down over instrumentation
            print(f"[efficiency] attach failed ({exc}); continuing uninstrumented")
            return _NullRecorder()
        return self

    def _attach(self):
        self._layers = find_decoder_layers(self.model)
        self._n_layers = len(self._layers)
        text_cfg = text_config_of(self.model)
        self._d = text_cfg.hidden_size
        self._m = text_cfg.intermediate_size

        for i, layer in enumerate(self._layers):
            self._handles.append(_register_pre_hook(layer, self._make_layer_pre_hook(i)))

        lm_head = find_lm_head(self.model)
        if lm_head is not None:
            self._handles.append(lm_head.register_forward_hook(self._lm_head_post_hook))

        vt = find_vision_tower(self.model)
        if vt is not None:
            self._handles.append(_register_pre_hook(vt, self._vision_pre_hook))
            self._handles.append(vt.register_forward_hook(self._vision_post_hook))
            vcfg = vision_config_of(vt)
            if vcfg is not None:
                self._vision_tflops = F.vision_tower_flops(vcfg) / 1e12
                proj = find_projector(self.model)
                if proj is not None:
                    patches = (vcfg.image_size // vcfg.patch_size) ** 2
                    self._vision_tflops += F.projector_flops(proj, patches) / 1e12

        if torch.cuda.is_available():
            self._weights_bytes = torch.cuda.memory_allocated()

        print(
            f"[efficiency] {self.method} mode={self.mode} run_id={self.run_id} "
            f"layers={self._n_layers} d={self._d} m={self._m} "
            f"vision={self._vision_tflops:.2f} TFLOPs -> {self._writer.path}"
        )

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ---------------------------------------------------------------- hooks (read-only)

    def _make_layer_pre_hook(self, idx):
        def hook(module, args, kwargs=None):
            try:
                hs = args[0] if args else (kwargs or {}).get("hidden_states")
                if hs is None or not hasattr(hs, "shape") or hs.dim() < 2:
                    return None
                n = hs.shape[1]
                if idx == 0:
                    self._on_forward_start(n)
                if self._forward_idx == 1:
                    # only the prefill pass carries a meaningful per-layer trace;
                    # decode passes are all n=1
                    self._trace.append(int(n))
                    if self.bench and idx == 0:
                        self._mark("prefill_llm_start")
            except Exception:
                pass
            return None

        return hook

    def _on_forward_start(self, n):
        self._forward_idx += 1
        if n == 1:
            self._decode_forwards += 1

    def _lm_head_post_hook(self, module, args, output):
        try:
            if self.bench and self._forward_idx == 1 and "prefill_end" not in self._events:
                self._mark("prefill_end")
        except Exception:
            pass
        return None

    def _vision_pre_hook(self, module, args, kwargs=None):
        try:
            if self.bench and "vision_start" not in self._events:
                self._mark("vision_start")
        except Exception:
            pass
        return None

    def _vision_post_hook(self, module, args, output):
        try:
            if self.bench and "vision_end" not in self._events:
                self._mark("vision_end")
        except Exception:
            pass
        return None

    # ---------------------------------------------------------------- timing

    def _mark(self, name):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._events[name] = ev

    def _elapsed(self, a, b):
        if a not in self._events or b not in self._events:
            return None
        try:
            return self._events[a].elapsed_time(self._events[b])
        except Exception:
            return None

    # ---------------------------------------------------------------- sample lifecycle

    def should_stop(self):
        """Lets `bench` runs stop after a subset without editing question files. Always False in
        `count` mode, so accuracy runs are never truncated."""
        return bool(self.limit) and self._written >= self.limit

    @contextmanager
    def sample(self, question_id=None):
        if not self.enabled:
            yield
            return
        self._begin()
        try:
            yield
        finally:
            try:
                self._end(question_id)
            except Exception as exc:
                print(f"[efficiency] sample record failed ({exc})")

    def _begin(self):
        self._trace = []
        self._forward_idx = 0
        self._decode_forwards = 0
        self._events = {}
        if self.bench and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            # Baseline per sample, not once at attach: whatever is resident right now (weights and
            # anything else) is the floor this generate() builds on, so the delta is exactly this
            # call's activation + KV footprint and can never come out negative.
            self._mem_baseline = torch.cuda.memory_allocated()
            self._mark("gen_start")
        self._wall_start = time.perf_counter()

    def _end(self, question_id):
        wall_ms = (time.perf_counter() - self._wall_start) * 1000.0
        idx = self._sample_idx
        self._sample_idx += 1

        if self.bench and torch.cuda.is_available():
            self._mark("gen_end")
            torch.cuda.synchronize()  # single sync per sample; resolves every mark at once

        if idx < self.warmup:
            return  # CUDA context init, autotune and cudnn benchmark land here
        if not self._trace:
            return

        gen_tokens = self._decode_forwards + 1
        prompt_tokens = self._trace[0]
        row = {
            "run_id": self.run_id,
            "method": self.method,
            "dataset": self.dataset,
            "mode": self.mode,
            "question_id": question_id,
            "config": self.config,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "total_tokens": prompt_tokens + gen_tokens,
            "layer_token_trace": self._trace,
            "equiv_prefill_tokens": sum(self._trace) / len(self._trace),
            "prefill_tflops": F.llm_stack_flops(self._trace, self._d, self._m) / 1e12,
            "vision_tflops": self._vision_tflops,
            "kv_cache_mb": F.kv_cache_bytes(self._trace, text_config_of(self.model)) / 1024**2,
            "wall_ms": wall_ms,
        }
        if self.notes:
            row["notes"] = self.notes

        if self.bench:
            ttft = self._elapsed("gen_start", "prefill_end")
            decode_total = self._elapsed("prefill_end", "gen_end")
            row.update({
                "ttft_ms": ttft,
                "vision_ms": self._elapsed("vision_start", "vision_end"),
                "prefill_ms": self._elapsed("prefill_llm_start", "prefill_end"),
                "decode_ms_per_tok": (
                    decode_total / max(gen_tokens - 1, 1) if decode_total is not None else None
                ),
                "decode_total_ms": decode_total,
            })
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated()
                row["peak_mem_mb"] = peak / 1024**2
                # absolute peak is ~14GB of fp16 weights and barely moves between methods;
                # the delta is the number that actually reflects pruning
                row["peak_mem_delta_mb"] = (peak - self._mem_baseline) / 1024**2

        self._writer.write(row)
        self._written += 1

    def close(self):
        self.detach()
        if self._writer is not None:
            self._writer.close(self._summary())

    def _summary(self):
        return {
            "run_id": self.run_id,
            "method": self.method,
            "dataset": self.dataset,
            "mode": self.mode,
            "config": self.config,
            "n_samples": self._written,
            "n_layers": self._n_layers,
            "hidden_size": self._d,
            "intermediate_size": self._m,
            "vision_tflops": self._vision_tflops,
            "weights_mb": self._weights_bytes / 1024**2,
            "env": _env_fingerprint(self.model),
        }


def _env_fingerprint(model):
    """Recorded on every run because latency is only comparable within a transformers/weights
    family: the forks here vendor 4.31.0, 4.37.2 and 4.39.0.dev0, and 4.31 has no SDPA for Llama
    at all. `attn_impl` also exposes FastV's eager-attention penalty (it needs output_attentions
    at its pruning layer), which is the main reason FLOPs and latency disagree.
    """
    info = {}
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    cfg = getattr(model, "config", None)
    if cfg is not None:
        info["attn_impl"] = getattr(cfg, "_attn_implementation", None)
        info["model_type"] = getattr(cfg, "model_type", None)
    proj = find_projector(model)
    if proj is not None:
        try:
            info["projector_dtype"] = str(next(proj.parameters()).dtype)
        except Exception:
            pass
    return info
