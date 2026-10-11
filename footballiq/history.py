"""Historical explorer: big five leagues, 2017-18 to 2021-22, with passing and pressing data.

The live system describes 2025-26 players with the few statistics FBref still
publishes. This module works on the multi-season corpus instead, where every
player-season has 83 features covering passing, carrying, pressing, shot creation
and touches by zone, and where the similarity metric has been learnt from how
players resemble themselves from one season to the next.

    similar_seasons("Kevin De Bruyne", season=2022)   who played like him that season
    style_timeline("Joshua Kimmich")                  how his profile moved year to year
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import corpus
from .linkage import find_by_name, normalise_name
from .metric import LinearIdentityMetric

EMBEDDING_DIM = 32


class HistoryStore:
    def __init__(self):
        self.table = corpus.outfield(corpus.load_seasons())
        features = corpus.feature_table(self.table)
        self.feature_names = list(features.columns)
        self.X = corpus.Standardiser.fit(features.to_numpy()).transform(features.to_numpy())
        self.ids = self.table["player_id"].to_numpy(object)
        self.season = self.table["season"].to_numpy()
        self.keys = self.table["player"].map(normalise_name).to_numpy(object)
        self.family = self.table["tm_position"].map(corpus.POSITION_FAMILY).to_numpy(object)
        # style metric learnt from player identity across seasons (fitted on every player:
        # the held-out evaluation of this model is in evaluation/benchmark_corpus.py)
        self.metric = LinearIdentityMetric(EMBEDDING_DIM).fit(self.X, self.ids)
        self.Z = self.metric.transform(self.X)
        sample = np.random.default_rng(0).choice(len(self.Z), size=min(1500, len(self.Z)), replace=False)
        d = np.linalg.norm(self.Z[sample][:, None, :] - self.Z[sample][None, :, :], axis=2)
        self.scale = float(np.percentile(d[np.triu_indices_from(d, 1)], 99.5))

    def find(self, name: str, season: int | None = None) -> int | None:
        """Row of a player-season; the player's latest season if ``season`` is not given."""
        row = find_by_name(name, self.keys, self.table["Min"].to_numpy())
        if row is None:
            return None
        mine = np.flatnonzero(self.ids == self.ids[row])
        if season is not None:
            mine = mine[self.season[mine] == season]
            return int(mine[0]) if len(mine) else None
        return int(mine[np.argmax(self.season[mine])])

    def similarity_from(self, row: int) -> np.ndarray:
        dist = np.linalg.norm(self.Z - self.Z[row], axis=1)
        return np.clip(1.0 - dist / self.scale, 0.0, 1.0)

    def shared_features(self, i: int, j: int, top: int = 3) -> list[str]:
        both = np.minimum(self.X[i], self.X[j])
        return [self.feature_names[k] for k in np.argsort(-both)[:top] if both[k] > 0.5]


@lru_cache(maxsize=1)
def get_history() -> HistoryStore:
    return HistoryStore()


def similar_seasons(name: str, season: int | None = None, top_n: int = 10, same_position: bool = True,
                    same_season: bool = False, max_age: int | None = None,
                    max_value_eur: float | None = None, store: HistoryStore | None = None) -> list[dict]:
    """Player-seasons most similar to ``name`` in ``season`` (other players only).

    ``same_season`` restricts candidates to the target's own season, which is what a
    club looking for a replacement at that time could actually have signed.
    ``max_value_eur`` keeps players whose Transfermarkt value was at most that amount.
    """
    store = store or get_history()
    row = store.find(name, season)
    if row is None:
        return []
    sims = store.similarity_from(row)
    table = store.table
    eligible = store.ids != store.ids[row]
    if same_position:
        eligible &= store.family == store.family[row]
    if same_season:
        eligible &= store.season == store.season[row]
    if max_age is not None:
        eligible &= table["age"].to_numpy() <= max_age
    if max_value_eur is not None:
        eligible &= table["market_value_eur"].fillna(np.inf).to_numpy() <= max_value_eur
    order = np.argsort(-np.where(eligible, sims, -1.0))
    out, seen = [], set()
    for j in order:
        if not eligible[j] or len(out) == top_n:
            break
        if store.ids[j] in seen:                       # one season per player in the list
            continue
        seen.add(store.ids[j])
        r = table.iloc[j]
        out.append({"player": r["player"], "season": int(r["season"]), "squad": r["squad"],
                    "position": r["tm_position"], "age": int(r["age"]) if r["age"] == r["age"] else None,
                    "minutes": int(r["Min"]),
                    "market_value_eur": None if pd.isna(r["market_value_eur"]) else float(r["market_value_eur"]),
                    "similarity": float(sims[j]), "shared": store.shared_features(row, j)})
    return out


def style_timeline(name: str, store: HistoryStore | None = None) -> list[dict]:
    """One entry per qualified season of a player, with how far his profile moved.

    ``change`` is the distance to his previous season as a percentile of the distance
    between two different players in the same position group: values near 0 mean he
    played the same way, values near 100 mean he changed as much as a different player would.
    """
    store = store or get_history()
    row = store.find(name)
    if row is None:
        return []
    mine = np.flatnonzero(store.ids == store.ids[row])
    mine = mine[np.argsort(store.season[mine])]
    peers = np.flatnonzero(store.family == store.family[row])
    rng = np.random.default_rng(0)
    a, b = rng.choice(peers, 4000), rng.choice(peers, 4000)
    reference = np.linalg.norm(store.Z[a] - store.Z[b], axis=1)[store.ids[a] != store.ids[b]]
    out = []
    for k, i in enumerate(mine):
        r = store.table.iloc[i]
        entry = {"season": int(r["season"]), "squad": r["squad"], "position": r["tm_position"],
                 "minutes": int(r["Min"]),
                 "market_value_eur": None if pd.isna(r["market_value_eur"]) else float(r["market_value_eur"]),
                 "change": None, "rose": [], "fell": []}
        if k > 0 and store.season[i] == store.season[mine[k - 1]] + 1:
            prev = mine[k - 1]
            entry["change"] = float(np.mean(reference < np.linalg.norm(store.Z[i] - store.Z[prev])) * 100)
            delta = store.X[i] - store.X[prev]
            entry["rose"] = [store.feature_names[j] for j in np.argsort(-delta)[:3] if delta[j] > 0.5]
            entry["fell"] = [store.feature_names[j] for j in np.argsort(delta)[:3] if delta[j] < -0.5]
        out.append(entry)
    return out


def _money(value) -> str:
    return "-" if value is None else f"EUR {value / 1e6:.1f}m"


def print_similar_seasons(name: str, season: int | None = None, top_n: int = 10, **filters) -> None:
    store = get_history()
    row = store.find(name, season)
    if row is None:
        print("Player (or that season) not found in the 2017-22 corpus.")
        return
    target = store.table.iloc[row]
    print(f"\nMost similar to {target['player']} in {target['season'] - 1}-{str(target['season'])[2:]} "
          f"({target['squad']}, {target['tm_position']}):\n")
    for rank, r in enumerate(similar_seasons(name, int(target["season"]), top_n, **filters), 1):
        print(f"{rank:>2}. {r['player']:<24} {r['season'] - 1}-{str(r['season'])[2:]}  {r['squad']:<18} "
              f"{_money(r['market_value_eur']):>11}  {r['similarity']:.3f}  shared: {', '.join(r['shared']) or '-'}")


def print_style_timeline(name: str) -> None:
    timeline = style_timeline(name)
    if not timeline:
        print("Player not found in the 2017-22 corpus.")
        return
    print(f"\n{'Season':<9}{'Club':<20}{'Minutes':>8}{'Value':>13}  {'Change':>6}  What moved")
    for e in timeline:
        change = "" if e["change"] is None else f"{e['change']:.0f}"
        moved = "; ".join(filter(None, ["up: " + ", ".join(e["rose"]) if e["rose"] else "",
                                         "down: " + ", ".join(e["fell"]) if e["fell"] else ""]))
        print(f"{e['season'] - 1}-{str(e['season'])[2:]:<4}{e['squad']:<20}{e['minutes']:>8}"
              f"{_money(e['market_value_eur']):>13}  {change:>6}  {moved}")


if __name__ == "__main__":
    player = input("Enter player name: ")
    print_similar_seasons(player)
    print_style_timeline(player)
