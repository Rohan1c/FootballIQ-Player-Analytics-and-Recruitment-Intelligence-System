"""Global scouting: find look-alikes outside the five leagues that have match statistics.

The EA database covers about 16,000 players worldwide, but FBref statistics exist
only for the big five European leagues. Because the embedding can be computed from
the attribute view alone (the missing performance view is predicted from it), every
EA player can be placed in the same style space as the fully observed players.

How much is lost by embedding from attributes only is measured in
``evaluation/benchmark_embeddings.py`` (the "missing view" experiment): players
held out from training are embedded both ways and the two embeddings are compared.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config, features
from .similarity import POSITION_FAMILY
from .store import PlayerStore, get_store


@lru_cache(maxsize=2)
def _pool(store: PlayerStore) -> tuple[pd.DataFrame, np.ndarray]:
    """EA outfield players that are not already in the linked (big-five) table."""
    ea = pd.read_csv(config.EA_RAW)
    linked_ids = set(store.players["player_id"])
    pool = ea[~ea["player_id"].isin(linked_ids) & ea["Position"].ne("GK")]
    pool = pool.dropna(subset=features.ATTRIBUTE_FEATURES).reset_index(drop=True)
    attributes = features.attribute_view(pool, store.views.stats)
    return pool, store.embedder.transform(performance=None, attributes=attributes)


def global_lookalikes(name: str, top_n: int = 10, max_age: int | None = None,
                      leagues: list[str] | None = None, same_position: bool = True,
                      store: PlayerStore | None = None) -> list[dict]:
    """Players from the worldwide EA pool who are closest in style to ``name``.

    The target is embedded from his attributes only as well, so that target and
    candidates are compared on equal terms.
    """
    store = store or get_store()
    i = store.find(name)
    if i is None or store.is_gk[i]:
        return []
    pool, z_pool = _pool(store)
    target_attr = store.views.attributes[[store._local[i]]]
    z_target = store.embedder.transform(performance=None, attributes=target_attr)[0]
    dist = np.linalg.norm(z_pool - z_target, axis=1)
    store._distances("OUT")                                   # make sure the scale exists
    sims = np.clip(1.0 - dist / store._scale["OUT"], 0.0, 1.0)

    eligible = np.ones(len(pool), dtype=bool)
    if max_age is not None:
        eligible &= pool["Age"].to_numpy() <= max_age
    if leagues:
        eligible &= pool["League"].isin(leagues).to_numpy()
    if same_position:
        family = pool["Position"].map(POSITION_FAMILY).to_numpy(object)
        eligible &= family == POSITION_FAMILY[store.positions[i]]
    order = np.argsort(-np.where(eligible, sims, -1.0))[:top_n]
    return [{
        "player": pool.at[j, "Name"], "team": pool.at[j, "Team"], "league": pool.at[j, "League"],
        "position": pool.at[j, "Position"], "age": int(pool.at[j, "Age"]),
        "similarity": float(sims[j]),
    } for j in order if eligible[j]]


def print_global_lookalikes(name: str, top_n: int = 10, max_age: int | None = None) -> None:
    store = get_store()
    i = store.find(name)
    if i is None:
        print("Player not found.")
        return
    results = global_lookalikes(name, top_n, max_age)
    age_note = f", aged {max_age} or under" if max_age else ""
    print(f"\nPlayers outside the big five leagues most similar to {store.names[i]}{age_note}:\n")
    for rank, r in enumerate(results, 1):
        team = r["team"] if isinstance(r["team"], str) else "-"
        league = r["league"] if isinstance(r["league"], str) else "-"
        print(f"{rank:>2}. {r['player']:<26} {r['position']:<4} age {r['age']:<3} "
              f"{team:<24} {league:<22} {r['similarity']:.3f}")


if __name__ == "__main__":
    print_global_lookalikes(input("Enter player name: "), top_n=10)
