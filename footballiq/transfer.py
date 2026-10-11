"""Replacement recommendations for a given player.

Score (weights unchanged from the original engine):

    0.40 * style similarity + 0.30 * role match + 0.20 * position match + 0.10 * age fit
"""
from __future__ import annotations

import numpy as np

from .compatibility import DEFAULT_MIN_MINUTES
from .store import PlayerStore, get_store

ROLE_FAMILIES = [
    {"Deep Playmaker", "Creative Playmaker", "Ball Winner", "Box-to-Box"},
    {"Wide Winger", "Creative Winger", "Inside Forward", "Poacher", "Target Forward", "False 9"},
    {"Ball Playing Defender", "Defensive Defender"},
]
_NEAR_POSITIONS = [({"CM", "CAM"}, 0.80), ({"RW", "RM"}, 0.80), ({"LW", "LM"}, 0.80),
                   ({"CDM", "CM"}, 0.75)]


def role_match(role_a: str, role_b: str) -> float:
    if role_a == role_b:
        return 1.0
    if any(role_a in family and role_b in family for family in ROLE_FAMILIES):
        return 0.75
    return 0.25


def position_match(pos_a: str, pos_b: str) -> float:
    if pos_a == pos_b:
        return 1.0
    for group, score in _NEAR_POSITIONS:
        if pos_a in group and pos_b in group:
            return score
    return 0.20


def replacement_row(i: int, store: PlayerStore | None = None) -> np.ndarray:
    """Replacement score of every player for player ``i``."""
    store = store or get_store()
    role = np.array([role_match(store.roles[i], r) for r in store.roles])
    position = np.array([position_match(store.positions[i], q) for q in store.positions])
    age = np.clip(1.0 - np.abs(store.ages - store.ages[i]) / 15.0, 0.0, None)
    return 0.40 * store.similarity_row(i) + 0.30 * role + 0.20 * position + 0.10 * age


def find_replacements(name: str, top_n: int = 10, younger_only: bool = False,
                      max_age: int | None = None, min_minutes: int = DEFAULT_MIN_MINUTES,
                      store: PlayerStore | None = None) -> list[dict]:
    """Ranked replacement candidates (empty list if the player is not found)."""
    store = store or get_store()
    i = store.find(name)
    if i is None:
        return []
    scores = replacement_row(i, store)
    eligible = (store.is_gk == store.is_gk[i]) & (store.minutes >= min_minutes)
    if younger_only:
        eligible &= store.ages < store.ages[i]
    if max_age is not None:
        eligible &= store.ages <= max_age
    eligible[i] = False
    order = np.argsort(-np.where(eligible, scores, -1.0))[:top_n]
    return [{"index": int(j), "player": store.names[j], "squad": store.players.at[j, "Squad"],
             "position": store.positions[j], "role": store.roles[j],
             "age": int(store.ages[j]), "score": float(scores[j])}
            for j in order if eligible[j]]


def print_replacements(name: str, top_n: int = 10, younger_only: bool = False) -> None:
    results = find_replacements(name, top_n, younger_only)
    if not results:
        print("Player not found." if get_store().find(name) is None else "No candidates.")
        return
    print("\n" + "=" * 70)
    print("BEST REPLACEMENTS")
    print("=" * 70)
    for rank, r in enumerate(results, 1):
        print(f"{rank}. {r['player']} | {r['position']} | {r['role']} | "
              f"Age {r['age']} | {r['score']:.3f}")


if __name__ == "__main__":
    target = input("Enter player name: ")
    younger = input("Only younger replacements? (y/n): ").strip().lower() == "y"
    print_replacements(target, younger_only=younger)
