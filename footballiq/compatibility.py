"""Tactical compatibility between two players.

The scoring formula is the original one (weights unchanged):

    0.40 * style similarity + 0.25 * role synergy + 0.15 * positional fit
    + 0.20 * statistical complementarity

What changed is how it is computed: one vectorised pass over all players instead of
a Python loop that re-classified every candidate's role on every call, and roles now
come from the shared store, so every engine sees the same role for the same player.

These weights are a design choice, not a fitted model: there is no ground truth for
"chemistry" in the data, so this score is not validated in the benchmark.
"""
from __future__ import annotations

import numpy as np

from .store import PlayerStore, get_store

DEFAULT_MIN_MINUTES = 900     # candidates need a meaningful sample of minutes

ROLE_SYNERGY = {
    "Creative Playmaker": ["Ball Winner", "Box-to-Box", "Deep Playmaker"],
    "Deep Playmaker": ["Ball Winner", "Box-to-Box", "Creative Playmaker"],
    "Ball Winner": ["Creative Playmaker", "Deep Playmaker", "Box-to-Box"],
    "Box-to-Box": ["Creative Playmaker", "Deep Playmaker", "Ball Winner"],
    "Creative Winger": ["Target Forward", "Poacher", "False 9"],
    "Wide Winger": ["Target Forward", "Poacher"],
    "Inside Forward": ["Creative Playmaker", "False 9", "Target Forward"],
    "False 9": ["Creative Winger", "Inside Forward"],
    "Poacher": ["Creative Winger", "Creative Playmaker"],
    "Target Forward": ["Creative Winger", "Inside Forward"],
    "Ball Playing Defender": ["Deep Playmaker", "Ball Winner"],
    "Defensive Defender": ["Ball Playing Defender"],
}
POSITION_GROUPS = {
    "CB": "DEF", "RB": "DEF", "LB": "DEF", "CDM": "MID", "CM": "MID", "CAM": "MID",
    "RW": "ATT", "LW": "ATT", "ST": "ATT", "RM": "ATT", "LM": "ATT", "GK": "GK",
}
_GROUP_FIT = {
    frozenset(("MID", "ATT")): 1.00, frozenset(("MID", "DEF")): 0.95,
    frozenset(("DEF", "ATT")): 0.70,
}


def role_compatibility(role_a: str, role_b: str) -> float:
    if role_a == role_b:
        return 0.30
    if role_b in ROLE_SYNERGY.get(role_a, ()) or role_a in ROLE_SYNERGY.get(role_b, ()):
        return 1.00
    return 0.50


def position_fit(pos_a: str, pos_b: str) -> float:
    group_a, group_b = POSITION_GROUPS.get(pos_a, "OTHER"), POSITION_GROUPS.get(pos_b, "OTHER")
    if group_a == group_b:
        return 0.60
    return _GROUP_FIT.get(frozenset((group_a, group_b)), 0.50)


def _tables(store: PlayerStore) -> dict:
    """Per-player arrays used by the vectorised score (built once per store)."""
    if not hasattr(store, "_compat_tables"):
        p = store.players
        roles = sorted(set(store.roles))
        positions = sorted(set(store.positions))
        store._compat_tables = {
            "creativity": (p["Vision"] + p["Passing"] + p["Ball Control"]).to_numpy(float) / 3,
            "defending": (p["Defending"] + p["Interceptions"] + p["Standing Tackle"]).to_numpy(float) / 3,
            "attack": (p["Finishing"] + p["Dribbling"] + p["Pace"]).to_numpy(float) / 3,
            "role_id": np.array([roles.index(r) for r in store.roles]),
            "role_matrix": np.array([[role_compatibility(a, b) for b in roles] for a in roles]),
            "pos_id": np.array([positions.index(q) for q in store.positions]),
            "pos_matrix": np.array([[position_fit(a, b) for b in positions] for a in positions]),
        }
    return store._compat_tables


def compatibility_row(i: int, store: PlayerStore | None = None) -> np.ndarray:
    """Compatibility of player ``i`` with every player in the store."""
    store = store or get_store()
    t = _tables(store)
    cre, dfn, att = t["creativity"], t["defending"], t["attack"]
    stat = np.maximum.reduce([cre[i] * dfn, cre * dfn[i], att[i] * cre, att * cre[i]]) / 10000.0
    return (
        0.40 * store.similarity_row(i)
        + 0.25 * t["role_matrix"][t["role_id"][i], t["role_id"]]
        + 0.15 * t["pos_matrix"][t["pos_id"][i], t["pos_id"]]
        + 0.20 * np.minimum(stat, 1.0)
    )


def compatibility_score(i: int, j: int, store: PlayerStore | None = None) -> float:
    return round(float(compatibility_row(i, store)[j]), 4)


def best_partners(name: str, top_n: int = 10, min_minutes: int = DEFAULT_MIN_MINUTES,
                  store: PlayerStore | None = None) -> list[dict]:
    """Outfield players who complement ``name`` best (empty list if not found)."""
    store = store or get_store()
    i = store.find(name)
    if i is None:
        return []
    scores = compatibility_row(i, store)
    eligible = ~store.is_gk & (store.minutes >= min_minutes)
    eligible[i] = False
    order = np.argsort(-np.where(eligible, scores, -1.0))[:top_n]
    return [{"index": int(j), "player": store.names[j], "position": store.positions[j],
             "role": store.roles[j], "score": float(scores[j])} for j in order if eligible[j]]


def print_best_partners(name: str, top_n: int = 10) -> None:
    store = get_store()
    i = store.find(name)
    if i is None:
        print("Player not found.")
        return
    print("\n" + "=" * 70)
    print(f"BEST PARTNERS FOR {store.names[i]}")
    print(f"ROLE: {store.roles[i]}")
    print("=" * 70)
    for r in best_partners(name, top_n):
        print(f"{r['player']:<30}{r['position']:<8}{r['role']:<25}{r['score']:.3f}")


if __name__ == "__main__":
    print_best_partners(input("Enter player name: "))
