"""Label-free and label-based metrics for player embeddings.

Every function takes an embedding matrix whose rows are players and returns a plain
float, so that any representation (raw features, PCA, an autoencoder) can be scored
with exactly the same code.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def neighbours(Z: np.ndarray, k: int = 10) -> np.ndarray:
    """Indices of the k nearest neighbours of every row (Euclidean, self excluded)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    return nn.kneighbors(Z, return_distance=False)[:, 1:]


def neighbour_purity(Z, labels, k: int = 10) -> float:
    """Share of the k nearest neighbours that carry the same label as the query."""
    labels = np.asarray(labels, object)
    return float((labels[neighbours(Z, k)] == labels[:, None]).mean())


def probe_macro_f1(Z, labels, seed: int = 0) -> float:
    """Macro-F1 of a linear classifier predicting ``labels`` from the embedding (5-fold CV)."""
    labels = np.asarray(labels).astype(str)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(f1_score(labels, cross_val_predict(model, Z, labels, cv=cv), average="macro"))


def leakage_r2(Z, target, seed: int = 0) -> float:
    """Cross-validated R^2 of a ridge regression predicting ``target`` from the embedding.

    Used with log-minutes as the target: a style embedding should not reveal how much
    a player played, so lower is better.
    """
    target = np.asarray(target, float)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    pred = cross_val_predict(model, Z, target, cv=KFold(5, shuffle=True, random_state=seed))
    return float(1.0 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def neighbour_gap(Z, values, k: int = 10) -> float:
    """Mean absolute difference in ``values`` between a player and his k neighbours."""
    values = np.asarray(values, float)
    return float(np.abs(values[neighbours(Z, k)] - values[:, None]).mean())


def same_group_share(Z, groups, k: int = 10) -> float:
    """Share of the k nearest neighbours that belong to the same group (e.g. club)."""
    return neighbour_purity(Z, groups, k)


def retrieval_map(Z, labels) -> float:
    """Mean average precision of same-label retrieval among labelled rows only."""
    labels = np.asarray(labels, object)
    dist = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    scores = []
    for i in range(len(Z)):
        order = np.argsort(dist[i])[:-1]
        hits = labels[order] == labels[i]
        if hits.sum() == 0:
            continue
        precision = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        scores.append(float((precision * hits).sum() / hits.sum()))
    return float(np.mean(scores))


def loo_centroid_predictions(Z, labels, allowed=None) -> np.ndarray:
    """Leave-one-out nearest-centroid classification.

    The centroid of every class is recomputed without the held-out player, so the
    player being classified never contributes to his own prototype. ``allowed`` is an
    optional list (one entry per row) of the classes permitted for that row.
    """
    labels = np.asarray(labels, object)
    classes = np.unique(labels)
    out = np.empty(len(Z), dtype=object)
    for i in range(len(Z)):
        keep = np.arange(len(Z)) != i
        best, best_dist = None, np.inf
        for c in classes:
            if allowed is not None and c not in allowed[i]:
                continue
            members = keep & (labels == c)
            if not members.any():
                continue
            dist = np.linalg.norm(Z[i] - Z[members].mean(axis=0))
            if dist < best_dist:
                best, best_dist = c, dist
        out[i] = best
    return out


def jaccard_stability(embeddings: list, k: int = 10) -> float:
    """Mean Jaccard overlap of k-neighbour sets between embeddings from different seeds."""
    sets = [neighbours(Z, k) for Z in embeddings]
    scores = []
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            inter = np.array([len(set(x) & set(y)) for x, y in zip(sets[a], sets[b])])
            scores.append(float(np.mean(inter / (2 * k - inter))))
    return float(np.mean(scores))


def bootstrap_ci(correct, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean of a 0/1 (or real) vector."""
    correct = np.asarray(correct, float)
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), size=(n_boot, len(correct)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
