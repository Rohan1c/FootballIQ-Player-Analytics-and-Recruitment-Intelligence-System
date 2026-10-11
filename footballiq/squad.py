"""Squad construction: best midfield trio and a 4-3-3 starting XI around a player.

The scoring rules are the original ones. Two behaviours were fixed:

* ``build_xi`` now puts the anchor player in the line-up at his own position. The
  original only did so for midfielders, so an XI "built around" a striker or a
  defender did not contain him.
* The goalkeeper is chosen by goalkeeping ability. Outfield chemistry is not
  defined for goalkeepers (their EA columns store goalkeeping skills).
"""
from __future__ import annotations

import numpy as np

from .compatibility import DEFAULT_MIN_MINUTES, compatibility_row
from .store import PlayerStore, get_store

MIDFIELD_ROLES = ["Creative Playmaker", "Deep Playmaker", "Ball Winner", "Box-to-Box"]
MIDFIELD_TEMPLATES = {
    "Creative Playmaker": ["Ball Winner", "Box-to-Box", "Deep Playmaker"],
    "Deep Playmaker": ["Creative Playmaker", "Ball Winner", "Box-to-Box"],
    "Ball Winner": ["Creative Playmaker", "Deep Playmaker", "Box-to-Box"],
    "Box-to-Box": ["Creative Playmaker", "Ball Winner", "Deep Playmaker"],
}
QUALITY_ATTRIBUTES = {
    "Creative Playmaker": ["Vision", "Passing", "Ball Control", "Dribbling"],
    "Deep Playmaker": ["Vision", "Passing", "Ball Control", "Crossing"],
    "Ball Winner": ["Defending", "Interceptions", "Standing Tackle", "Aggression"],
    "Box-to-Box": ["Physicality", "Pace", "Passing", "Dribbling"],
}
POOL_SIZE = 15


def role_fit(anchor_role: str, role: str) -> float:
    if role in MIDFIELD_TEMPLATES.get(anchor_role, []):
        return 1.0
    if role == anchor_role:
        return 0.50
    return 0.25 if role in MIDFIELD_ROLES else 0.0


def midfield_quality(store: PlayerStore, i: int) -> float:
    columns = QUALITY_ATTRIBUTES.get(store.roles[i])
    return float(store.players.loc[i, columns].sum()) / 400.0 if columns else 0.0


def best_trio(anchor, min_minutes: int = DEFAULT_MIN_MINUTES, store: PlayerStore | None = None):
    """Best two midfield partners for ``anchor`` (a name or a row index).

    Returns ``((anchor, a, b), score)`` or ``(None, 0)`` if the anchor is not a
    midfielder by role.
    """
    store = store or get_store()
    i = store.find(anchor) if isinstance(anchor, str) else anchor
    if i is None or store.roles[i] not in MIDFIELD_ROLES:
        return None, 0
    anchor_role = store.roles[i]
    with_anchor = compatibility_row(i, store)
    midfield = np.isin(store.positions, ["CDM", "CM", "CAM"]) & (store.minutes >= min_minutes)
    midfield[i] = False

    pools = {}
    for role in MIDFIELD_ROLES:
        members = np.flatnonzero(midfield & (store.roles == role))
        pools[role] = members[np.argsort(-with_anchor[members])][:POOL_SIZE]
    rows = {int(j): compatibility_row(int(j), store) for pool in pools.values() for j in pool}
    quality = {j: midfield_quality(store, j) for j in rows}

    best, best_score = None, -1.0
    for role_a in MIDFIELD_ROLES:
        for role_b in MIDFIELD_ROLES:
            if role_a == role_b:
                continue
            fit_a, fit_b = role_fit(anchor_role, role_a), role_fit(anchor_role, role_b)
            for a in map(int, pools[role_a]):
                for b in map(int, pools[role_b]):
                    chemistry = (with_anchor[a] + with_anchor[b] + rows[a][b]) / 3
                    tactical = 0.60 * (fit_a + fit_b) / 2 + 0.40 * chemistry
                    score = (0.50 * tactical + 0.20 * (quality[a] + quality[b]) / 2
                             + 0.15 * fit_a + 0.15 * fit_b)
                    if score > best_score:
                        best, best_score = (i, a, b), float(score)
    return best, best_score


def print_trio(trio, score, store: PlayerStore | None = None) -> None:
    store = store or get_store()
    if trio is None:
        print("No valid trio found (the anchor must be a midfielder).")
        return
    print("\nBEST TACTICAL MIDFIELD\n")
    for j in trio:
        print(f"{store.names[j]} | {store.positions[j]} | {store.roles[j]}")
    print(f"\nTactical Score: {score:.3f}")


# ---------------------------------------------------------------- starting XI
XI_ORDER = ["GK", "RB", "CB1", "CB2", "LB", "CDM", "CM1", "CM2", "RW", "ST", "LW"]
SLOT_POSITIONS = {
    "GK": ["GK"], "RB": ["RB"], "LB": ["LB"], "CB1": ["CB"], "CB2": ["CB"],
    "CDM": ["CDM"], "CM1": ["CM", "CAM"], "CM2": ["CM", "CAM"],
    "RW": ["RW", "RM"], "LW": ["LW", "LM"], "ST": ["ST"],
}
ROLE_BONUS = {
    "CDM": {"Ball Winner": 0.10, "Deep Playmaker": 0.08},
    "CM": {"Creative Playmaker": 0.10, "Box-to-Box": 0.08},
    "RW": {"Wide Winger": 0.10, "Creative Winger": 0.08},
    "LW": {"Inside Forward": 0.10, "Creative Winger": 0.08},
    "ST": {"Target Forward": 0.10, "Poacher": 0.08, "False 9": 0.08},
    "CB": {"Defensive Defender": 0.10, "Ball Playing Defender": 0.08},
}
GK_SKILLS = ["Pace", "Shooting", "Dribbling", "Physicality"]   # diving, handling, reflexes, positioning


def build_xi(anchor, min_minutes: int = DEFAULT_MIN_MINUTES, store: PlayerStore | None = None) -> dict:
    """4-3-3 around ``anchor`` (a name or a row index): slot -> row index."""
    store = store or get_store()
    i = store.find(anchor) if isinstance(anchor, str) else anchor
    if i is None:
        return {}
    chemistry = compatibility_row(i, store)
    eligible = store.minutes >= min_minutes
    xi, used = {}, {i}

    anchor_slot = next(s for s in XI_ORDER if store.positions[i] in SLOT_POSITIONS[s])
    xi[anchor_slot] = i
    for slot in XI_ORDER:
        if slot in xi:
            continue
        candidates = np.flatnonzero(np.isin(store.positions, SLOT_POSITIONS[slot]) & eligible)
        candidates = np.array([c for c in candidates if c not in used])
        if slot == "GK":
            score = store.players.loc[candidates, GK_SKILLS].mean(axis=1).to_numpy(float)
        else:
            bonus = ROLE_BONUS.get(slot.rstrip("12"), {})
            score = chemistry[candidates] + np.array([bonus.get(store.roles[c], 0.0) for c in candidates])
        pick = int(candidates[np.argmax(score)])
        xi[slot] = pick
        used.add(pick)
    return xi


def print_xi(xi: dict, store: PlayerStore | None = None) -> None:
    store = store or get_store()
    print("\n" + "=" * 70)
    print("RECOMMENDED 4-3-3")
    print("=" * 70)
    for slot in XI_ORDER:
        j = xi[slot]
        print(f"{slot:<5}{store.names[j]:<30}{store.positions[j]:<6}{store.roles[j]}")


if __name__ == "__main__":
    name = input("Enter player name: ")
    trio, trio_score = best_trio(name)
    print_trio(trio, trio_score)
    line_up = build_xi(name)
    if line_up:
        print_xi(line_up)
