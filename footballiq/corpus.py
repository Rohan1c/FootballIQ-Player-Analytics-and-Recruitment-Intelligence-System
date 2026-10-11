"""Multi-season corpus: big five leagues, 2017-18 to 2021-22, with event-derived statistics.

The 2025-26 FBref file used by the live system only has goals, shots, crosses, fouls,
tackles won and interceptions (FBref lost its advanced data in January 2026). This
corpus has what that file lacks - passing, carrying, pressing, shot creation and
touches by zone - for 13,229 player-seasons, with a stable player id that links a
player across seasons and to his Transfermarkt position, market value and transfers.

It is used for research (``evaluation/benchmark_corpus.py``) and for the historical
explorer (``footballiq/history.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .features import fbref_line, shrink_rate, shrink_ratio
from .linkage import normalise_name

QUALIFIED_MINUTES = 900

# ---------------------------------------------------------------- feature definitions
RATES = [      # season totals turned into shrunk per-90 rates
    # shooting
    "npG", "Sh", "SoT", "npxG",
    # chance creation and progression by passing
    "Ast", "xA", "KeyPasses", "PassFinalThird", "PassPenArea", "CrossPenArea", "PrgPasses",
    "ThroughBalls", "Switches", "Crosses", "Corners", "PassFK",
    # passing volume and type
    "PassAtt", "ShortAtt", "MediumAtt", "LongAtt", "PassUnderPressure", "PassHigh", "PassHead", "ThrowIns",
    # shot-creating actions
    "SCA", "SCADribble", "SCAShot", "SCAFouled", "SCADefence",
    # defending
    "Tackles", "TacklesWon", "DribblersFaced", "DribbledPast", "Pressures", "PressuresWon", "Blocks",
    "ShotsBlocked", "PassesBlocked", "Interceptions", "Clearances", "Recoveries", "AerialsWon",
    "AerialsLost", "Fouls",
    # carrying, receiving and ball security
    "Touches", "TouchesDefPen", "TouchesAttPen", "Dribbles", "DribblesWon", "Carries", "PrgCarries",
    "CarriesFinalThird", "CarriesPenArea", "Miscontrols", "Dispossessed", "PassesReceived",
    "PrgPassesReceived", "Fouled", "Offsides",
]
RATIOS = {     # successes / trials, shrunk towards the positional average
    "PassCompletion": ("PassCmp", "PassAtt"),
    "LongPassCompletion": ("LongCmp", "LongAtt"),
    "LongPassShare": ("LongAtt", "PassAtt"),
    "ShortPassShare": ("ShortAtt", "PassAtt"),
    "GroundPassShare": ("PassGround", "PassAtt"),
    "ShotAccuracy": ("SoT", "Sh"),
    "DribbleSuccess": ("DribblesWon", "Dribbles"),
    "DuelTackleSuccess": ("DribblersTackled", "DribblersFaced"),
    "PressureSuccess": ("PressuresWon", "Pressures"),
    "AerialSuccess": ("AerialsWon", "Aerials"),
    "ReceptionSuccess": ("PassesReceived", "PassTargets"),
    "TouchShareDef3rd": ("TouchesDef3rd", "Touches"),
    "TouchShareMid3rd": ("TouchesMid3rd", "Touches"),
    "TouchShareAtt3rd": ("TouchesAtt3rd", "Touches"),
    "PressureShareAtt3rd": ("PressuresAtt3rd", "Pressures"),
    "PressureShareDef3rd": ("PressuresDef3rd", "Pressures"),
    "TackleShareAtt3rd": ("TacklesAtt3rd", "Tackles"),
    "TackleShareDef3rd": ("TacklesDef3rd", "Tackles"),
}
AVERAGES = {   # total / events: an average per event, with a light prior
    "PassLength": ("PassDist", "PassCmp"),
    "PassProgressiveShare": ("PassPrgDist", "PassDist"),
    "CarryLength": ("CarryDist", "Carries"),
    "CarryProgressiveShare": ("CarryPrgDist", "CarryDist"),
    "ShotDistance": ("ShotDistTotal", "Sh"),
    "npxGPerShot": ("npxG", "Sh"),
}
AVERAGE_PRIOR = {"PassLength": 20, "PassProgressiveShare": 400, "CarryLength": 20,
                 "CarryProgressiveShare": 200, "ShotDistance": 10, "npxGPerShot": 10}

# The only columns the 2025-26 FBref file still has. Used to measure what the lost data costs.
THIN_RATES = ["npG", "Ast", "Sh", "SoT", "Crosses", "Fouled", "Fouls", "Offsides", "Interceptions",
              "TacklesWon", "CrdY"]
THIN_RATIOS = {"ShotAccuracy": ("SoT", "Sh"), "Conversion": ("npG", "Sh")}

POSITION_FAMILY = {
    "Centre-Back": "CB", "Left-Back": "FB", "Right-Back": "FB", "Defensive Midfield": "DM",
    "Central Midfield": "CM", "Attacking Midfield": "AM", "Left Midfield": "WIDE",
    "Right Midfield": "WIDE", "Left Winger": "WIDE", "Right Winger": "WIDE",
    "Second Striker": "ST", "Centre-Forward": "ST",
}


def load_seasons() -> pd.DataFrame:
    seasons = pd.read_csv(config.CORPUS_SEASONS)
    seasons["Aerials"] = seasons["AerialsWon"] + seasons["AerialsLost"]
    return seasons


def load_transfers() -> pd.DataFrame:
    return pd.read_csv(config.CORPUS_TRANSFERS)


def outfield(seasons: pd.DataFrame, min_minutes: int = QUALIFIED_MINUTES) -> pd.DataFrame:
    """Qualified outfield player-seasons (goalkeepers removed by either source)."""
    keep = (seasons["Min"] >= min_minutes) & ~seasons["pos"].fillna("").str.startswith("GK") \
        & seasons["tm_position"].ne("Goalkeeper") & seasons["tm_position"].notna()
    return seasons[keep].reset_index(drop=True)


def _shrink_average(total, events, groups, prior_events: float) -> np.ndarray:
    total, events = np.asarray(total, float), np.asarray(events, float)
    groups = np.asarray(groups, object)
    out = np.zeros_like(total)
    for g in np.unique(groups):
        m = groups == g
        mean = total[m].sum() / max(events[m].sum(), 1e-9)
        out[m] = (total[m] + prior_events * mean) / (events[m] + prior_events)
    return out


def feature_table(seasons: pd.DataFrame, thin: bool = False) -> pd.DataFrame:
    """Raw-scale features: shrunk per-90 rates, shrunk ratios and per-event averages.

    ``thin=True`` returns only what can be computed from the columns that survive in
    the 2025-26 FBref file.
    """
    nineties = seasons["Min"].to_numpy(float) / 90.0
    groups = seasons["pos"].map(fbref_line).to_numpy(object)
    rates, ratios = (THIN_RATES, THIN_RATIOS) if thin else (RATES, RATIOS)
    columns = {name: np.sqrt(shrink_rate(seasons[name], nineties, groups)) for name in rates}
    for name, (succ, trials) in ratios.items():
        columns[name] = shrink_ratio(seasons[succ], seasons[trials], groups)
    if not thin:
        for name, (total, events) in AVERAGES.items():
            columns[name] = _shrink_average(seasons[total], seasons[events], groups, AVERAGE_PRIOR[name])
    return pd.DataFrame(columns, index=seasons.index)


@dataclass
class Standardiser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardiser":
        std = values.std(axis=0)
        std[std < 1e-9] = 1.0
        return cls(values.mean(axis=0), std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def career_features(live_players: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rich statistics from each current player's most recent qualified corpus season.

    Players are linked on normalised name and birth year (ambiguous keys are skipped).
    Returns ``(has_history, H)``: a boolean mask over ``live_players`` and a matrix of
    standardised rich features, zero for players without history.
    """
    cohort = outfield(load_seasons())
    rich = feature_table(cohort).to_numpy()
    rich = Standardiser.fit(rich).transform(rich)
    latest: dict = {}
    owners: dict = {}
    for row, (name, born, season, pid) in enumerate(
            zip(cohort["player"], cohort["born"], cohort["season"], cohort["player_id"])):
        if born != born:
            continue
        key = (normalise_name(name), int(born))
        owners.setdefault(key, set()).add(pid)
        if key not in latest or season > cohort.at[latest[key], "season"]:
            latest[key] = row
    has_history = np.zeros(len(live_players), dtype=bool)
    H = np.zeros((len(live_players), rich.shape[1]))
    for i, (name, born) in enumerate(zip(live_players["Player"], live_players["Born"])):
        if born != born:
            continue
        key = (normalise_name(name), int(born))
        if key in latest and len(owners[key]) == 1:
            has_history[i] = True
            H[i] = rich[latest[key]]
    return has_history, H
