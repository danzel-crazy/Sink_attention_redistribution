"""Shared registry for the visualization modules.

Split out so each topic (sink tokens, visual attention, ...) can live in its own
module while all of them register into one registry that
`visualization/render_fastv_visualizations.py` reads. Importing this file has no side
effects and pulls in no heavy deps, so topic modules and the renderer can share it
freely.

A topic module:
  * imports `visualization` (the decorator) and `subdir` from here,
  * defines a module-level `SUBDIR = "<folder name>"`,
  * tags each plot with `@visualization` and writes its PNGs into
    `subdir(out_dir, SUBDIR)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

VISUALIZATION_REGISTRY: Dict[str, Callable] = {}


def visualization(fn: Callable) -> Callable:
    """Tag a function as a runnable visualization and register it by name."""
    fn.is_visualization = True
    VISUALIZATION_REGISTRY[fn.__name__] = fn
    return fn


def subdir(out_dir: Path, name: str) -> Path:
    """Return `out_dir/name`, creating it if needed.

    Topic modules call this so every PNG they emit lands in their own subfolder
    (e.g. `output/textvqa/sink_tokens/`) instead of the flat top-level out dir.
    """
    target = Path(out_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    return target
