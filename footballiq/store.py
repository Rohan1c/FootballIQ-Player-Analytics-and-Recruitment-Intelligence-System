"""Single source of truth for players, features, embeddings and roles.

Every engine reads from one shared :class:`PlayerStore`, built once per process by
:func:`get_store`. This replaces the original design in which each module re-read
the CSV, rebuilt its own 2057 x 2057 distance matrix at import time and recomputed
every player's role inside the query loop (about 11 s per query; see
``evaluation/benchmark_speed.py``).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config, features, linkage, roles
from .embedding import Embedder


class PlayerStore:
    def __init__(self, method: str = config.EMBEDDING_METHOD, dim: int = config.EMBEDDING_DIM,
                 include_draft_labels: bool = True, seed: int = config.SEED):
        if not config.PLAYERS_CSV.exists():
            linkage.build(verbose=False)
        self.players = pd.read_csv(config.PLAYERS_CSV)
        self.n = len(self.players)
        self.keys = self.players["Player"].map(linkage.normalise_name).to_numpy(object)
        self.names = self.players["Player"].to_numpy(object)
        self.positions = self.players["Position"].to_numpy(object)
        self.minutes = self.players["Min"].to_numpy(float)
        self.ages = self.players["Age"].to_numpy(float)
        self.is_gk = self.positions == "GK"

        # ---- outfield players: two views -> embedding
        self.views = features.build_views(self.players, config.MIN_MINUTES)
        self.outfield = self.views.index
        self.X = self.views.combined                       # (n_outfield, 44) standardised
        train = self.minutes[self.outfield] >= config.MIN_MINUTES
        self.embedder = Embedder(method, dim, seed).fit(
            self.views.performance[train], self.views.attributes[train])
        z_out = self.embedder.transform(self.views.performance, self.views.attributes)

        # ---- goalkeepers live in their own space
        self.goalkeepers, z_gk, self.gk_feature_names = features.goalkeeper_features(self.players)
        self._z = {"OUT": z_out, "GK": z_gk}
        self._members = {"OUT": self.outfield, "GK": self.goalkeepers}
        self._local = np.zeros(self.n, dtype=int)          # row of each player inside his space
        self._local[self.outfield] = np.arange(len(self.outfield))
        self._local[self.goalkeepers] = np.arange(len(self.goalkeepers))
        self._dist: dict[str, np.ndarray] = {}
        self._scale: dict[str, float] = {}

        # ---- roles
        self._fit_roles(include_draft_labels)

    # ------------------------------------------------------------------ lookup
    def find(self, name: str) -> int | None:
        """Row index of the best match for ``name`` (None if nothing is close).

        Exact normalised name, then whole words, then the start of a word, then a
        fuzzy match. Ties go to the player with the most minutes. (The original
        code took the first alphabetical substring hit, so "Rodri" returned
        Rodrigo Bentancur and "Son" returned Brenden Aaronson.)
        """
        return linkage.find_by_name(name, self.keys, self.minutes)

    def row(self, i: int) -> pd.Series:
        return self.players.iloc[i]

    def space(self, i: int) -> str:
        return "GK" if self.is_gk[i] else "OUT"

    def embedding(self, i: int) -> np.ndarray:
        return self._z[self.space(i)][self._local[i]]

    # ------------------------------------------------------------------ similarity
    def _distances(self, space: str) -> np.ndarray:
        if space not in self._dist:
            z = self._z[space].astype(np.float32)
            sq = (z ** 2).sum(axis=1)
            d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * z @ z.T, 0.0))
            np.fill_diagonal(d, 0.0)
            self._dist[space] = d
            # reference distance: almost every pair is closer than this
            self._scale[space] = float(np.percentile(d[np.triu_indices_from(d, 1)], 99.5))
        return self._dist[space]

    def similarity_row(self, i: int) -> np.ndarray:
        """Similarity in [0, 1] between player ``i`` and every player (0 across spaces)."""
        space = self.space(i)
        d = self._distances(space)[self._local[i]]
        out = np.zeros(self.n)
        out[self._members[space]] = np.clip(1.0 - d / self._scale[space], 0.0, 1.0)
        return out

    def similarity(self, i: int, j: int) -> float:
        return float(self.similarity_row(i)[j])

    # ------------------------------------------------------------------ roles
    def _fit_roles(self, include_draft: bool) -> None:
        labels = roles.load_labels(include_draft)
        lookup = {(p, b): i for i, (p, b) in enumerate(
            zip(self.players["Player"], self.players["Born"].astype(int)))}
        rows, targets = [], []
        for player, born, role in zip(labels["player"], labels["born"], labels["role"]):
            i = lookup.get((player, int(born)))
            if i is not None and not self.is_gk[i]:
                rows.append(self._local[i])
                targets.append(role)
        self.role_training_rows = np.array(rows)
        self.role_classifier = roles.RoleClassifier().fit(self.X[rows], targets)
        self.role_classes = self.role_classifier.classes_
        self.role_proba = self.role_classifier.predict_proba(self.X, self.positions[self.outfield])
        primary = np.asarray(self.role_classes, object)[self.role_proba.argmax(axis=1)]
        self.roles = np.full(self.n, roles.GOALKEEPER, dtype=object)
        self.roles[self.outfield] = primary
        # a player who has a human label keeps it; the model fills in everyone else
        self.roles[self.outfield[self.role_training_rows]] = targets

    def role_scores(self, i: int) -> dict:
        """Role probabilities for player ``i`` (roles his position does not allow are omitted)."""
        if self.is_gk[i]:
            return {roles.GOALKEEPER: 1.0}
        proba = self.role_proba[self._local[i]]
        return {r: float(p) for r, p in zip(self.role_classes, proba) if p > 0}


@lru_cache(maxsize=4)
def get_store(method: str = config.EMBEDDING_METHOD, dim: int = config.EMBEDDING_DIM,
              include_draft_labels: bool = True) -> PlayerStore:
    """Shared store (built on first use, then reused by every engine)."""
    return PlayerStore(method, dim, include_draft_labels)
