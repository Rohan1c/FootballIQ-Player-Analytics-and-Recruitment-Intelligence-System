"""Feature engineering: two views of every outfield player.

* performance view  - what the player did on the pitch (FBref), as per-90 rates
* attribute view    - how the player is rated by scouts (EA FC detailed attributes)

Design decisions, each addressing a measured weakness of the original pipeline:

1. Per-90 rates instead of season totals. Totals (goals, minutes, starts) made the
   original embedding encode *how much* a player played rather than *how* he plays.
2. Empirical-Bayes shrinkage of every rate. A player with 90 minutes and one goal
   does not score 1.0 goals per 90; his rate is pulled towards the average of his
   positional group in proportion to how little he has played (Gamma-Poisson model
   for counts, Beta-Binomial for ratios).
3. Availability and team-result columns (minutes, starts, points per match, +/-)
   are not style features and are kept out of the representation.
4. The six EA "face" ratings (Pace, Shooting, ...) are weighted sums of the detailed
   attributes, so they are dropped to avoid counting the same information twice.
5. Goalkeepers are handled separately: for them EA stores goalkeeping skills in the
   outfield columns, so mixing them with outfield players is meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- column groups
RATE_COUNTS = {            # feature name -> FBref count column (per 90 minutes)
    "npGoals90": "G-PK",
    "Assists90": "Ast",
    "Shots90": "Sh",
    "ShotsOnTarget90": "SoT",
    "Crosses90": "Crs",
    "FoulsDrawn90": "Fld",
    "FoulsCommitted90": "Fls",
    "Offsides90": "Off",
    "Interceptions90": "Int",
    "TacklesWon90": "TklW",
    "YellowCards90": "CrdY",
}
RATIOS = {                 # feature name -> (successes, trials)
    "ShotAccuracy": ("SoT", "Sh"),
    "Conversion": ("G-PK", "Sh"),
}
PERFORMANCE_FEATURES = list(RATE_COUNTS) + list(RATIOS)

ATTRIBUTE_FEATURES = [
    "Acceleration", "Sprint Speed",
    "Positioning", "Finishing", "Shot Power", "Long Shots", "Volleys", "Penalties",
    "Vision", "Crossing", "Free Kick Accuracy", "Short Passing", "Long Passing", "Curve",
    "Agility", "Balance", "Reactions", "Ball Control", "Composure", "Dribbling",
    "Interceptions", "Heading Accuracy", "Def Awareness", "Standing Tackle", "Sliding Tackle",
    "Jumping", "Stamina", "Strength", "Aggression",
    "Height", "Weight",
]
FACE_RATINGS = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physicality"]

# EA re-uses the six face columns for goalkeeping skills.
GK_ATTRIBUTES = {
    "Pace": "GK Diving", "Shooting": "GK Handling", "Passing": "GK Kicking",
    "Dribbling": "GK Reflexes", "Defending": "GK Speed", "Physicality": "GK Positioning",
}

POSITION_GROUP = {
    "GK": "GK", "CB": "DEF", "RB": "DEF", "LB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID",
    "RM": "WIDE", "LM": "WIDE", "RW": "WIDE", "LW": "WIDE", "ST": "FWD",
}


def fbref_line(pos: str) -> str:
    """Primary FBref line (DF / MF / FW / GK) - the first tag FBref lists."""
    return str(pos).split(",")[0].strip()


# ---------------------------------------------------------------- shrinkage
def shrink_rate(count, exposure, groups, min_exposure: float = 10.0):
    """Empirical-Bayes (Gamma-Poisson) estimate of a per-90 rate.

    count ~ Poisson(rate * exposure), rate ~ Gamma(alpha, beta) within each group.
    The posterior mean is (count + alpha) / (exposure + beta): the prior acts like
    ``beta`` extra 90s played at the group-average rate. alpha and beta are fitted
    by moment matching on players with at least ``min_exposure`` 90s.
    """
    count = np.asarray(count, float)
    exposure = np.asarray(exposure, float)
    groups = np.asarray(groups, object)
    out = np.zeros_like(count)
    for g in np.unique(groups):
        m = groups == g
        ref = m & (exposure >= min_exposure)
        if ref.sum() < 20:                      # tiny group: fall back to everyone
            ref = exposure >= min_exposure
        mean = count[ref].sum() / exposure[ref].sum()
        rates = count[ref] / exposure[ref]
        sampling = mean * np.mean(1.0 / exposure[ref])       # Poisson noise in the rates
        between = max(rates.var() - sampling, 1e-6)          # true spread between players
        beta = np.clip(mean / between, 1.0, 40.0) if mean > 0 else 40.0
        out[m] = (count[m] + mean * beta) / (exposure[m] + beta)
    return out


def shrink_ratio(successes, trials, groups, min_trials: float = 15.0):
    """Empirical-Bayes (Beta-Binomial) estimate of a success ratio."""
    successes = np.asarray(successes, float)
    trials = np.asarray(trials, float)
    groups = np.asarray(groups, object)
    out = np.zeros_like(successes)
    for g in np.unique(groups):
        m = groups == g
        ref = m & (trials >= min_trials)
        if ref.sum() < 20:
            ref = trials >= min_trials
        mean = successes[ref].sum() / trials[ref].sum()
        ratios = successes[ref] / trials[ref]
        sampling = mean * (1 - mean) * np.mean(1.0 / trials[ref])
        between = max(ratios.var() - sampling, 1e-6)
        strength = np.clip(mean * (1 - mean) / between - 1.0, 2.0, 200.0)
        out[m] = (successes[m] + mean * strength) / (trials[m] + strength)
    return out


# ---------------------------------------------------------------- views
@dataclass
class Views:
    """Standardised feature matrices for outfield players (rows aligned with ``index``)."""
    index: np.ndarray                 # row positions in the players table
    performance: np.ndarray           # (n, 13)
    attributes: np.ndarray            # (n, 31)
    performance_names: list
    attribute_names: list
    stats: dict                       # means / stds used for standardisation

    @property
    def combined(self) -> np.ndarray:
        return np.hstack([self.performance, self.attributes])

    @property
    def names(self) -> list:
        return self.performance_names + self.attribute_names


def performance_table(players: pd.DataFrame) -> pd.DataFrame:
    """Shrunk per-90 rates and ratios (raw scale, one row per player)."""
    nineties = players["Min"].to_numpy(float) / 90.0
    groups = players["Pos"].map(fbref_line).to_numpy(object)
    out = pd.DataFrame(index=players.index)
    for name, col in RATE_COUNTS.items():
        out[name] = shrink_rate(players[col], nineties, groups)
    for name, (succ, trials) in RATIOS.items():
        out[name] = shrink_ratio(players[succ], players[trials], groups)
    return out


def _standardise(frame: pd.DataFrame, fit_rows: np.ndarray, stats: dict | None, key: str):
    values = frame.to_numpy(float)
    if stats is None or key not in stats:
        mean = values[fit_rows].mean(axis=0)
        std = values[fit_rows].std(axis=0)
        std[std < 1e-9] = 1.0
        stats = dict(stats or {})
        stats[key] = (mean, std)
    mean, std = stats[key]
    return (values - mean) / std, stats


def build_views(players: pd.DataFrame, min_minutes: int = 450, stats: dict | None = None) -> Views:
    """Build both views for all outfield players.

    Rates are square-root transformed (variance stabilising for counts) and then
    z-scored using players with at least ``min_minutes`` so that low-sample players
    do not distort the scale.
    """
    outfield = players["Position"].ne("GK").to_numpy()
    index = np.flatnonzero(outfield)
    sub = players.iloc[index]
    perf = performance_table(players).iloc[index]
    perf[list(RATE_COUNTS)] = np.sqrt(perf[list(RATE_COUNTS)])
    attr = sub[ATTRIBUTE_FEATURES].astype(float)
    fit_rows = sub["Min"].to_numpy() >= min_minutes
    P, stats = _standardise(perf, fit_rows, stats, "performance")
    A, stats = _standardise(attr, fit_rows, stats, "attributes")
    return Views(index, P, A, list(perf.columns), list(attr.columns), stats)


def attribute_view(frame: pd.DataFrame, stats: dict) -> np.ndarray:
    """Standardise EA attributes of players that have no FBref record."""
    mean, std = stats["attributes"]
    return (frame[ATTRIBUTE_FEATURES].to_numpy(float) - mean) / std


def goalkeeper_features(players: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Standardised goalkeeper profile (shot-stopping record + EA goalkeeping skills)."""
    index = np.flatnonzero(players["Position"].eq("GK").to_numpy())
    gk = players.iloc[index]
    nineties = np.maximum(gk["Min"].to_numpy(float) / 90.0, 1e-9)
    one_group = np.full(len(gk), "GK", dtype=object)
    table = pd.DataFrame({
        "SavePct": shrink_ratio(gk["Saves"], gk["SoTA"], one_group),
        "GoalsAgainst90": shrink_rate(gk["GA"], nineties, one_group),
        "ShotsFaced90": shrink_rate(gk["SoTA"], nineties, one_group),
        "CleanSheetPct": shrink_ratio(gk["CS"], gk["MP"], one_group, min_trials=10),
    }, index=gk.index)
    for col, name in GK_ATTRIBUTES.items():
        table[name] = gk[col].astype(float)
    table["Height"] = gk["Height"].astype(float)
    values = table.to_numpy(float)
    values = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-9)
    return index, values, list(table.columns)
