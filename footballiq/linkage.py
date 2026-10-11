"""Record linkage between FBref season statistics and EA FC player attributes.

The original pipeline joined the two sources on the exact player-name string. That
drops every player whose name is spelled differently in the two sources (accents,
nicknames such as "Vini Jr."), keeps a single club spell for players who moved
mid-season, and can attach the attributes of a different person with the same name.

This module replaces it with a three-pass linkage:

  0. FBref rows are aggregated per player (name + birth year) so that players with
     two club spells keep all of their minutes.
  1. Exact match on an accent/punctuation-normalised name, confirmed by club or,
     failing that, by nation (single-token nicknames always need the club).
  2. Within the same club, match on name-token containment or string similarity
     (catches "Gabriel" vs "Gabriel Magalhaes", "Vini Jr." vs "Vinicius Junior").
  3. Within the same nation, match on high string similarity (players whose EA
     record lists a different club, e.g. loans).

Every pass enforces two hard constraints: goalkeepers only match goalkeepers, and
the two sources must agree on age to within ``MAX_AGE_GAP`` years.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from . import config

MAX_AGE_GAP = 2

# FBref columns that are season totals and can be summed across club spells.
COUNT_COLS = [
    "MP", "Starts", "Min", "Subs", "Compl", "unSub",
    "Gls", "Ast", "G-PK", "PK", "PKatt", "CrdY", "CrdR", "2CrdY",
    "Sh", "SoT", "Fls", "Fld", "Off", "Crs", "Int", "TklW", "OG",
    "onG", "onGA",
    "GA", "SoTA", "Saves", "W", "D", "L", "CS", "PKA", "PKsv", "PKm",
]

_SPECIAL = str.maketrans({
    "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "ß": "ss",
    "æ": "ae", "Æ": "AE", "œ": "oe", "ı": "i", "þ": "th", "ð": "d",
    "’": "'", "`": "'",
})
_SUFFIXES = {"jr", "junior", "sr", "ii", "iii"}


def normalise_name(name: str) -> str:
    """Lower-case ASCII key: accents stripped, punctuation removed."""
    text = str(name).translate(_SPECIAL)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "").replace("-", " ")
    text = re.sub(r"[^a-z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(key: str) -> set[str]:
    return {t for t in key.split() if t not in _SUFFIXES}


def name_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two normalised names.

    The structural rules below score under 1.0 on purpose: they are only applied
    inside a block (same club, compatible age) where few candidates remain, and a
    match still has to beat the runner-up by a margin.
    """
    if a == b:
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    if ta <= tb or tb <= ta:                       # "gabriel" / "gabriel magalhaes"
        return 0.90

    def abbreviates(short: set[str], long_: set[str]) -> bool:
        return all(any(l.startswith(s) and len(s) >= 3 for l in long_) for s in short)

    if abbreviates(ta, tb) or abbreviates(tb, ta):   # "vini" / "vinicius", "emi" / "emiliano"
        return 0.88
    last_a, last_b = a.split()[-1], b.split()[-1]
    if last_a == last_b and len(last_a) >= 5:        # same surname, different given name
        return 0.84
    return SequenceMatcher(None, a, b).ratio()


def aggregate_fbref(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple club spells of one player into a single season row."""
    raw = raw.copy()
    raw["Born"] = raw["Born"].fillna(-1)
    counts = [c for c in COUNT_COLS if c in raw.columns]
    raw[counts] = raw[counts].fillna(0)

    keys = ["Player", "Born"]
    order = raw.sort_values("Min", ascending=False)
    grouped = order.groupby(keys, sort=False)
    main = grouped[["Nation", "Pos", "Squad", "Comp", "Age"]].first()   # spell with most minutes
    totals = grouped[counts].sum()
    squads = grouped["Squad"].agg(" / ".join).rename("Squads")
    spells = grouped.size().rename("n_spells")
    out = pd.concat([main, totals, squads, spells], axis=1).reset_index()
    out = out.sort_values("Player", kind="stable").reset_index(drop=True)
    out["is_gk"] = out["Pos"].astype(str).str.startswith("GK")
    out["key"] = out["Player"].map(normalise_name)
    return out


def _learn_mapping(pairs: pd.DataFrame, left: str, right: str, min_support: int = 3) -> dict:
    """Majority-vote mapping between two categorical columns, learnt from safe matches."""
    table = pairs.groupby([left, right]).size().reset_index(name="n")
    table = table.sort_values("n", ascending=False).drop_duplicates(left)
    return dict(table.loc[table["n"] >= min_support, [left, right]].values)


def link(fbref_raw: pd.DataFrame, ea_raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Return one row per FBref player with the matched EA attributes attached."""
    fb = aggregate_fbref(fbref_raw)
    ea = ea_raw.copy().reset_index(drop=True)
    ea["key"] = ea["Name"].map(normalise_name)
    ea["is_gk"] = ea["Position"].eq("GK")
    ea_age = ea["Age"].to_numpy(float)
    ea_gk = ea["is_gk"].to_numpy()
    ea_key = ea["key"].to_numpy(object)
    ea_team = ea["Team"].to_numpy(object)
    ea_nation = ea["Nation"].to_numpy(object)
    fb_key = fb["key"].to_numpy(object)
    fb_age = fb["Age"].to_numpy(float)
    fb_gk = fb["is_gk"].to_numpy()
    fb_squads = [[s.strip() for s in str(v).split("/")] for v in fb["Squads"]]
    mononym = np.array([len(k.split()) == 1 for k in fb_key])

    match = np.full(len(fb), -1)
    how = np.array([""] * len(fb), dtype=object)
    used = np.zeros(len(ea), dtype=bool)
    by_key = ea.groupby("key").indices

    def admissible(i: int, cand) -> np.ndarray:
        cand = np.asarray(cand, dtype=int)
        if len(cand) == 0:
            return cand
        ok = (ea_gk[cand] == fb_gk[i]) & (np.abs(ea_age[cand] - fb_age[i]) <= MAX_AGE_GAP)
        return cand[ok & ~used[cand]]

    def assign(i: int, j: int, method: str) -> None:
        match[i], how[i] = j, method
        used[j] = True

    # ---- bootstrap: unambiguous full-name matches teach us how the two sources
    #      spell clubs ("Lazio" vs "Latium") and nations ("us USA" vs "United States")
    seed_fb, seed_ea = [], []
    for i in range(len(fb)):
        cand = admissible(i, by_key.get(fb_key[i], []))
        if len(cand) == 1 and not mononym[i]:
            seed_fb.append(i)
            seed_ea.append(cand[0])
    safe = pd.DataFrame({
        "Squad": fb.loc[seed_fb, "Squad"].to_numpy(),
        "Team": ea_team[seed_ea],
        "fb_nation": fb.loc[seed_fb, "Nation"].to_numpy(),
        "ea_nation": ea_nation[seed_ea],
    }).dropna()
    club_map = _learn_mapping(safe, "Squad", "Team", min_support=5)
    nation_map = _learn_mapping(safe, "fb_nation", "ea_nation", min_support=2)
    by_team = ea.groupby("Team").indices
    by_nation = ea.groupby("Nation").indices
    fb_teams = [{club_map.get(s) for s in squads} - {None} for squads in fb_squads]
    fb_nation = [nation_map.get(n) for n in fb["Nation"]]

    # ---- pass 1: exact normalised name. Club agreement is accepted outright; without
    #      it the nations must agree and the name must not be a single-token nickname
    #      ("Wesley", "Lucas"), which is too ambiguous on its own.
    for require_club in (True, False):
        for i in np.flatnonzero(match < 0):
            cand = admissible(i, by_key.get(fb_key[i], []))
            if len(cand) == 0:
                continue
            same_club = cand[np.isin(ea_team[cand], list(fb_teams[i]))] if fb_teams[i] else cand[:0]
            if len(same_club):
                best = same_club[np.argmin(np.abs(ea_age[same_club] - fb_age[i]))]
                assign(i, best, "exact+club")
            elif not require_club and not mononym[i]:
                cand = cand[ea_nation[cand] == fb_nation[i]]
                if len(cand) == 1:
                    assign(i, cand[0], "exact+nation")

    def best_fuzzy(i: int, cand: np.ndarray, scorer, threshold: float, margin: float):
        sims = np.array([scorer(fb_key[i], ea_key[c]) for c in cand])
        order = np.argsort(-sims)
        runner_up = sims[order[1]] if len(sims) > 1 else 0.0
        if sims[order[0]] >= threshold and sims[order[0]] - runner_up >= margin:
            return cand[order[0]]
        return None

    # ---- pass 2: same club, fuzzy name
    for i in np.flatnonzero(match < 0):
        pools = [np.asarray(by_team.get(t, []), dtype=int) for t in fb_teams[i]]
        cand = admissible(i, np.concatenate(pools)) if pools else np.array([], dtype=int)
        if len(cand):
            j = best_fuzzy(i, cand, name_similarity, threshold=0.82, margin=0.05)
            if j is not None:
                assign(i, j, "club+fuzzy")

    # ---- pass 3: same nation, near-identical spelling (EA lists a different club)
    def ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    for i in np.flatnonzero((match < 0) & ~mononym):
        cand = admissible(i, by_nation.get(fb_nation[i], []))
        cand = np.array([c for c in cand if ea_key[c][:1] == fb_key[i][:1]], dtype=int)
        if len(cand):
            j = best_fuzzy(i, cand, ratio, threshold=0.92, margin=0.04)
            if j is not None:
                assign(i, j, "nation+fuzzy")

    fb["match_method"] = how
    fb["ea_index"] = match
    linked = fb[match >= 0].reset_index(drop=True)
    ea_cols = [c for c in ea.columns if c not in ("key", "is_gk", "page", "slug", "source_url")]
    attached = ea.loc[linked["ea_index"], ea_cols].reset_index(drop=True)
    attached = attached.rename(columns={"Age": "ea_age", "Nation": "ea_nation", "Name": "ea_name"})
    linked = pd.concat([linked.drop(columns=["ea_index", "key"]), attached], axis=1)
    linked["club_agrees"] = [
        t in {club_map.get(s.strip()) for s in str(squads).split("/")}
        for squads, t in zip(linked["Squads"], linked["Team"])
    ]

    if verbose:
        total = len(fb)
        print(f"FBref players (after merging club spells): {total}")
        print(f"linked: {len(linked)} ({len(linked) / total:.1%}), "
              f"minutes covered: {linked['Min'].sum() / fb['Min'].sum():.1%}")
        print(linked["match_method"].value_counts().to_string())
    return linked


def build(verbose: bool = True) -> pd.DataFrame:
    """Run the linkage on the raw files and write ``data/players_linked.csv``."""
    fbref = pd.read_csv(config.FBREF_RAW)
    ea = pd.read_csv(config.EA_RAW)
    linked = link(fbref, ea, verbose=verbose)
    linked.to_csv(config.PLAYERS_CSV, index=False)
    if verbose:
        print(f"wrote {config.PLAYERS_CSV.relative_to(config.ROOT)} ({len(linked)} players)")
    return linked


if __name__ == "__main__":
    build()
