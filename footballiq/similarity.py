"""Similar-player search with a plain-language explanation of each match."""
from __future__ import annotations

import numpy as np

from .store import PlayerStore, get_store

# Positions that count as "the same position" when filtering candidates.
POSITION_FAMILY = {
    "GK": "GK", "CB": "CB", "RB": "FB", "LB": "FB", "CDM": "CM", "CM": "CM",
    "CAM": "AM", "RM": "WIDE", "LM": "WIDE", "RW": "WIDE", "LW": "WIDE", "ST": "ST",
}


def explain(store: PlayerStore, i: int, j: int, top: int = 3) -> dict:
    """Which features two players share, and where they differ most.

    Values are z-scores against the players of the same kind (outfield players or
    goalkeepers), so "shared" lists features on which both are well above average.
    """
    if store.is_gk[i] != store.is_gk[j]:
        return {"shared": [], "different": []}
    if store.is_gk[i]:
        values, names = store._z["GK"], np.array(store.gk_feature_names, dtype=object)
    else:
        values, names = store.X, np.array(store.views.names, dtype=object)
    a, b = values[store._local[i]], values[store._local[j]]
    both = np.minimum(a, b)
    shared = [(names[k], float(a[k]), float(b[k])) for k in np.argsort(-both)[:top] if both[k] > 0.5]
    gap = np.abs(a - b)
    different = [(names[k], float(a[k]), float(b[k])) for k in np.argsort(-gap)[:top]]
    return {"shared": shared, "different": different}


def similar_players(name: str, top_n: int = 10, same_position: bool = True,
                    min_minutes: int = 0, store: PlayerStore | None = None) -> list[dict]:
    """Most similar players to ``name``.

    ``same_position`` restricts candidates to the player's position family (centre
    backs with centre backs, wide players with wide players, ...), as scouts do.
    """
    store = store or get_store()
    i = store.find(name)
    if i is None:
        return []
    sims = store.similarity_row(i)
    eligible = (store.is_gk == store.is_gk[i]) & (store.minutes >= min_minutes)
    if same_position:
        family = np.array([POSITION_FAMILY[p] for p in store.positions], dtype=object)
        eligible &= family == family[i]
    eligible[i] = False
    order = np.argsort(-np.where(eligible, sims, -1.0))[:top_n]
    out = []
    for j in order:
        if not eligible[j]:
            break
        row = store.row(j)
        out.append({
            "index": int(j), "player": row["Player"], "squad": row["Squad"],
            "position": row["Position"], "age": int(row["Age"]), "minutes": int(row["Min"]),
            "role": store.roles[j], "similarity": float(sims[j]),
            "low_sample": bool(row["Min"] < 450),
            "why": explain(store, i, j),
        })
    return out


def get_similar_players(player_name: str, top_n: int = 5) -> list[tuple[str, float]]:
    """Backwards-compatible wrapper: prints the list and returns (name, score) pairs."""
    store = get_store()
    i = store.find(player_name)
    if i is None:
        print("Player not found!")
        return []
    results = similar_players(player_name, top_n)
    print(f"\nMatched Player: {store.names[i]}")
    print(f"\nTop players similar to {store.names[i]}:\n")
    for r in results:
        shared = ", ".join(n for n, _, _ in r["why"]["shared"]) or "-"
        flag = "  (low minutes)" if r["low_sample"] else ""
        print(f"{r['player']:<28} {r['similarity']:.3f}   shared: {shared}{flag}")
    return [(r["player"], r["similarity"]) for r in results]


if __name__ == "__main__":
    get_similar_players(input("Enter player name: "), top_n=10)
