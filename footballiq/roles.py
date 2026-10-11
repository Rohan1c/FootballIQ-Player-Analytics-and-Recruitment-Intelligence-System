"""Tactical role classification.

The original engine scored roles with hand-written bonus and penalty weights that
were tuned against the same 46 players used to report its accuracy. This module
replaces those weights with a model fitted to labelled examples:

* a regularised multinomial logistic regression on the 44 standardised features,
* followed by the *position gate* of the original engine (a goalkeeper cannot be
  a Poacher), applied as a hard prior on the predicted probabilities.

Nothing is tuned by hand, so its accuracy can be estimated honestly by
cross-validation (see ``evaluation/benchmark_roles.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config

ROLES = [
    "Deep Playmaker", "Creative Playmaker", "Ball Winner", "Box-to-Box",
    "Wide Winger", "Creative Winger", "Inside Forward",
    "Poacher", "Target Forward", "False 9",
    "Ball Playing Defender", "Defensive Defender",
]
GOALKEEPER = "Goalkeeper"

# Roles a player may be assigned given his listed position.
ORIGINAL_POSITION_ROLE_MAP = {
    "GK": [GOALKEEPER],
    "CB": ["Ball Playing Defender", "Defensive Defender"],
    "RB": ["Ball Playing Defender", "Wide Winger"],
    "LB": ["Ball Playing Defender", "Wide Winger"],
    "CDM": ["Deep Playmaker", "Ball Winner", "Box-to-Box"],
    "CM": ["Deep Playmaker", "Ball Winner", "Box-to-Box", "Creative Playmaker"],
    "CAM": ["Creative Playmaker", "False 9", "Box-to-Box"],
    "RW": ["Wide Winger", "Creative Winger", "Inside Forward"],
    "LW": ["Wide Winger", "Creative Winger", "Inside Forward"],
    "RM": ["Wide Winger", "Creative Winger", "Box-to-Box"],
    "LM": ["Wide Winger", "Creative Winger", "Box-to-Box"],
    "ST": ["Poacher", "Target Forward", "False 9"],
}
# One change from the original gate: EA lists most wingers as RM/LM (290 players
# against 81 RW/LW), and the original gate did not allow an RM/LM to be an Inside
# Forward. That made the correct role unreachable for players such as Jarrod Bowen.
POSITION_ROLE_MAP = {
    **ORIGINAL_POSITION_ROLE_MAP,
    "RM": ORIGINAL_POSITION_ROLE_MAP["RM"] + ["Inside Forward"],
    "LM": ORIGINAL_POSITION_ROLE_MAP["LM"] + ["Inside Forward"],
}

REGULARISATION = 0.1      # inverse strength C of the logistic regression


def load_labels(include_draft: bool = True) -> pd.DataFrame:
    """Role labels: the project's own list plus (optionally) the unverified draft set."""
    frames = [pd.read_csv(config.ROLE_LABELS)]
    if include_draft and config.ROLE_LABELS_DRAFT.exists():
        frames.append(pd.read_csv(config.ROLE_LABELS_DRAFT))
    labels = pd.concat(frames, ignore_index=True)
    return labels[labels["role"].isin(ROLES)].reset_index(drop=True)


def gate_matrix(positions, classes, gate=None) -> np.ndarray:
    """Boolean matrix: is role ``classes[j]`` allowed for a player at ``positions[i]``?"""
    gate = POSITION_ROLE_MAP if gate is None else gate
    classes = list(classes)
    out = np.zeros((len(positions), len(classes)), dtype=bool)
    for i, pos in enumerate(positions):
        allowed = gate.get(pos, classes)
        out[i] = [c in allowed for c in classes]
    return out


def apply_gate(proba: np.ndarray, positions, classes, gate=None) -> np.ndarray:
    """Zero out roles the position does not allow and renormalise."""
    gated = proba * gate_matrix(positions, classes, gate)
    total = gated.sum(axis=1, keepdims=True)
    return np.where(total > 0, gated / np.maximum(total, 1e-12), proba)


class RoleClassifier:
    """Logistic regression over the feature matrix, gated by position."""

    def __init__(self, C: float = REGULARISATION, use_gate: bool = True, gate=None):
        self.C, self.use_gate, self.gate = C, use_gate, gate

    def fit(self, X: np.ndarray, roles) -> "RoleClassifier":
        self.model = LogisticRegression(C=self.C, max_iter=5000).fit(X, np.asarray(roles, object))
        self.classes_ = list(self.model.classes_)
        return self

    def predict_proba(self, X: np.ndarray, positions=None) -> np.ndarray:
        proba = self.model.predict_proba(X)
        if self.use_gate and positions is not None:
            proba = apply_gate(proba, positions, self.classes_, self.gate)
        return proba

    def predict(self, X: np.ndarray, positions=None) -> np.ndarray:
        proba = self.predict_proba(X, positions)
        return np.asarray(self.classes_, object)[proba.argmax(axis=1)]


class CareerAwareRoleClassifier:
    """Role classifier that also uses a player's earlier seasons when they are known.

    The 2025-26 statistics contain no passing or pressing data. About 40% of current
    outfield players also appear in the 2017-22 corpus, which does. For those players a
    second classifier is trained on the current features plus the rich statistics of
    their latest corpus season; everyone else is handled by the base classifier.

    This is an experiment, not the shipped classifier: in cross-validation it adds about
    one point of accuracy, which is inside the noise (``evaluation/benchmark_roles.py``).
    Statistics that are four years old are a weak substitute for current ones.
    """

    MIN_HISTORY_LABELS = 60

    def __init__(self, C: float = REGULARISATION, gate=None):
        self.C, self.gate = C, gate

    def fit(self, X, H, has_history, roles) -> "CareerAwareRoleClassifier":
        roles = np.asarray(roles, object)
        has_history = np.asarray(has_history, bool)
        self.base = RoleClassifier(self.C, gate=self.gate).fit(X, roles)
        self.classes_ = self.base.classes_
        self.career = None
        if has_history.sum() >= self.MIN_HISTORY_LABELS:
            both = np.hstack([X[has_history], H[has_history]])
            self.career = RoleClassifier(self.C, gate=self.gate).fit(both, roles[has_history])
        return self

    def predict_proba(self, X, H, has_history, positions=None) -> np.ndarray:
        has_history = np.asarray(has_history, bool)
        proba = self.base.predict_proba(X, positions)
        if self.career is not None and has_history.any():
            rows = np.flatnonzero(has_history)
            sub = self.career.predict_proba(np.hstack([X[rows], H[rows]]),
                                            None if positions is None else np.asarray(positions, object)[rows])
            aligned = np.zeros((len(rows), len(self.classes_)))
            for j, role in enumerate(self.career.classes_):
                aligned[:, self.classes_.index(role)] = sub[:, j]
            proba[rows] = aligned
        return proba

    def predict(self, X, H, has_history, positions=None) -> np.ndarray:
        proba = self.predict_proba(X, H, has_history, positions)
        return np.asarray(self.classes_, object)[proba.argmax(axis=1)]
