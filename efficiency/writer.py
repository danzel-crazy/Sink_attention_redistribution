"""Per-sample JSONL + a run summary beside it.

One row per sample rather than an aggregate, so a suspicious mean can always be traced back to
the samples that produced it. The summary carries the run's config and environment fingerprint,
which is what makes rows self-describing months later.
"""

import json
import os


DEFAULT_DIR = os.environ.get(
    "EFFICIENCY_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "efficiency", "runs"),
)


class RunWriter:
    def __init__(self, path=None, run_id="run"):
        if path is None:
            os.makedirs(DEFAULT_DIR, exist_ok=True)
            path = os.path.join(DEFAULT_DIR, f"{run_id}.jsonl")
        else:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.path = path
        self._fh = open(path, "w")

    def write(self, row):
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()  # runs get killed; partial data still beats none

    def close(self, summary=None):
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
        if summary is not None:
            with open(self.path.replace(".jsonl", ".summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
