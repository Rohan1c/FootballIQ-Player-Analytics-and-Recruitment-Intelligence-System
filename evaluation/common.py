"""Shared loading code for the benchmark scripts."""
from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from footballiq import config, features

RESULTS = config.RESULTS_DIR
POSITION_FAMILY = {      # side-agnostic position: left/right is not a style property
    "CB": "CB", "RB": "FB", "LB": "FB", "CDM": "CDM", "CM": "CM", "CAM": "CAM",
    "RM": "WIDE", "LM": "WIDE", "RW": "WIDE", "LW": "WIDE", "ST": "ST",
}
QUALIFIED_MINUTES = 900


def load_new():
    """Linked players, their two views, and the outfield sub-table (rows aligned with the views)."""
    players = pd.read_csv(config.PLAYERS_CSV)
    views = features.build_views(players, config.MIN_MINUTES)
    outfield = players.iloc[views.index].reset_index(drop=True)
    return players, views, outfield


def legacy_feature_names() -> list[str]:
    source = (config.LEGACY_DIR / "models" / "autoencoder.py").read_text(encoding="utf-8")
    return ast.literal_eval(re.search(r"selected_features = (\[.*?\])", source, re.S).group(1))


def load_legacy():
    """Original merged table, its 63 standardised features and the shipped 64-d embedding."""
    table = pd.read_csv(config.LEGACY_MERGED).drop_duplicates("Player").reset_index(drop=True)
    cols = legacy_feature_names()
    X = StandardScaler().fit_transform(table[cols].fillna(table[cols].median()))
    return table, X, np.load(config.LEGACY_EMBEDDINGS)


def common_cohort(outfield: pd.DataFrame, legacy: pd.DataFrame):
    """Outfield players with 900+ minutes that exist in both the original and the new table.

    Returns (mask over the new outfield rows, matching row indices in the legacy table).
    """
    legacy_row = {name: i for i, name in enumerate(legacy["Player"])}
    ambiguous = outfield["Player"].duplicated(keep=False)
    mask = np.array([
        (name in legacy_row) and minutes >= QUALIFIED_MINUTES and not dup
        for name, minutes, dup in zip(outfield["Player"], outfield["Min"], ambiguous)
    ])
    return mask, np.array([legacy_row[name] for name in outfield["Player"][mask]])


def role_labels(outfield: pd.DataFrame, include_draft: bool):
    """(row indices into the outfield table, role labels, source) for labelled outfield players."""
    frames = [pd.read_csv(config.ROLE_LABELS)]
    if include_draft:
        frames.append(pd.read_csv(config.ROLE_LABELS_DRAFT))
    labels = pd.concat(frames, ignore_index=True)
    labels = labels[labels["position"] != "GK"]
    lookup = {(p, int(b)): i for i, (p, b) in enumerate(zip(outfield["Player"], outfield["Born"]))}
    rows = np.array([lookup[(p, int(b))] for p, b in zip(labels["player"], labels["born"])])
    return rows, labels["role"].to_numpy(object), labels["source"].to_numpy(object)


def to_markdown(frame: pd.DataFrame, digits: int = 3) -> str:
    """Small dependency-free Markdown table writer."""
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value)
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])
