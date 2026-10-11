"""Query latency: original engines vs the shared-store implementation.

Each side runs in a fresh Python process so that import-time work is counted.
Run from the repository root:  python -m evaluation.benchmark_speed
"""
from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

from evaluation import common
from footballiq import config

LEGACY = r"""
import json, time
out = {}
t = time.perf_counter()
import legacy.models.similarity_engine as S
import legacy.models.compatibility_engine as C
import legacy.models.transfer_engine as T
out["start-up (load data, build matrices)"] = time.perf_counter() - t
for name, fn in [("similar players", lambda: S.get_similar_players("Pedri", 10)),
                 ("best partners", lambda: C.best_partners("Pedri", 10)),
                 ("replacements", lambda: T.find_replacements("Pedri", 10))]:
    t = time.perf_counter(); fn(); out[name] = time.perf_counter() - t
print()
print(json.dumps(out))
"""
NEW = r"""
import json, time
out = {}
t = time.perf_counter()
from footballiq.store import get_store
from footballiq.similarity import similar_players
from footballiq.compatibility import best_partners
from footballiq.transfer import find_replacements
store = get_store(); store.similarity_row(0)
out["start-up (load data, build matrices)"] = time.perf_counter() - t
for name, fn in [("similar players", lambda: similar_players("Pedri", 10)),
                 ("best partners", lambda: best_partners("Pedri", 10)),
                 ("replacements", lambda: find_replacements("Pedri", 10))]:
    runs = []
    for _ in range(20):
        t = time.perf_counter(); fn(); runs.append(time.perf_counter() - t)
    out[name] = sorted(runs)[len(runs) // 2]
print(json.dumps(out))
"""


def _run(code: str) -> dict:
    done = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=config.ROOT,
                          capture_output=True, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> pd.DataFrame:
    legacy, new = _run(LEGACY), _run(NEW)
    table = pd.DataFrame({"operation": list(legacy), "original_seconds": list(legacy.values()),
                          "new_seconds": [new[k] for k in legacy]})
    table["speed_up"] = table["original_seconds"] / table["new_seconds"]
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(common.RESULTS / "speed.csv", index=False)
    print("\n== query latency ==")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table


if __name__ == "__main__":
    main()
