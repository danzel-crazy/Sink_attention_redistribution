"""Global efficiency counting for the token-pruning methods in this repo.

Usage from any eval loader (see efficiency/PLAN.md for the design and its caveats):

    from efficiency import EfficiencyRecorder

    recorder = EfficiencyRecorder.from_env(model, method="sparsevlm", config={...})
    ...
    for ... in loader:
        if recorder.should_stop():
            break
        with recorder.sample(question_id=idx):
            output_ids = model.generate(...)
    recorder.close()

`from_env` returns a no-op unless EFFICIENCY_MODE is set, so runs are unchanged by default.
"""

from .recorder import EfficiencyRecorder

__all__ = ["EfficiencyRecorder"]
