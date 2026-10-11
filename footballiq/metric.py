"""Learning a style metric from player identity across seasons.

With several seasons per player, "the same player a year later" is a free training
signal: a good style space should place a player's seasons close together and other
players further away. Two models use it, in the way speaker-verification systems learn
from several recordings of the same speaker:

* :class:`LinearIdentityMetric` - linear discriminant analysis with players as classes.
  It rescales the feature space by the within-player scatter, so directions that
  fluctuate from season to season for the same player count for less.
* :class:`ContrastiveEncoder` - a small neural network trained with the InfoNCE loss:
  for a batch of players, each player's season must pick out his own other season
  among everyone else's.

Both are trained on one set of players and evaluated on players they have never seen
(``evaluation/benchmark_corpus.py``). The linear model is the baseline the neural one
has to beat.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _pairs_by_player(player_ids) -> dict:
    index: dict = {}
    for row, pid in enumerate(player_ids):
        index.setdefault(pid, []).append(row)
    return {pid: np.array(rows) for pid, rows in index.items() if len(rows) >= 2}


class LinearIdentityMetric:
    """LDA over player identity with a shrunk within-player covariance."""

    def __init__(self, dim: int = 32, shrinkage: float = 0.1):
        self.dim, self.shrinkage = dim, shrinkage

    def fit(self, X: np.ndarray, player_ids) -> "LinearIdentityMetric":
        X = np.asarray(X, float)
        groups = _pairs_by_player(player_ids)
        self.mean_ = X.mean(axis=0)
        within = np.zeros((X.shape[1], X.shape[1]))
        n = 0
        for rows in groups.values():
            centred = X[rows] - X[rows].mean(axis=0)
            within += centred.T @ centred
            n += len(rows) - 1
        within /= max(n, 1)
        target = np.eye(X.shape[1]) * np.trace(within) / X.shape[1]
        within = (1 - self.shrinkage) * within + self.shrinkage * target
        total = np.cov(X - self.mean_, rowvar=False)
        # generalised eigenproblem: directions with the most total variance per unit of
        # within-player variance; eigenvectors are scaled so within-player covariance is I
        values, vectors = eigh(total, within)
        order = np.argsort(values)[::-1][: self.dim]
        self.projection_ = vectors[:, order]
        self.ratios_ = values[order]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.mean_) @ self.projection_


class ContrastiveEncoder:
    """MLP encoder trained with a symmetric InfoNCE loss on (player, other season) pairs.

    Parameters
    ----------
    dim : size of the (L2-normalised) embedding.
    hidden : hidden-layer widths.
    temperature : softmax temperature of the contrastive loss.
    noise, feature_drop : input augmentation (Gaussian noise in z-score units and the
        share of features set to the average) applied independently to both seasons.
    """

    def __init__(self, dim=32, hidden=(256, 128), temperature=0.1, noise=0.1, feature_drop=0.1,
                 lr=1e-3, weight_decay=1e-4, batch_size=128, epochs=300, patience=30, seed=0):
        self.dim, self.hidden, self.temperature = dim, tuple(hidden), temperature
        self.noise, self.feature_drop = noise, feature_drop
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.epochs, self.patience, self.seed = epochs, patience, seed
        self.history_: list[tuple[float, float]] = []

    # ------------------------------------------------------------------ network
    def _init(self, n_in: int, rng: np.random.Generator) -> None:
        sizes = [n_in, *self.hidden, self.dim]
        self.weights = [rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros(b) for b in sizes[1:]]

    def _forward(self, x: np.ndarray):
        activations = [x]
        h = x
        for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if layer < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
            activations.append(h)
        norm = np.linalg.norm(h, axis=1, keepdims=True) + 1e-12
        return activations, h / norm, norm

    def _loss_and_grads(self, anchors: np.ndarray, positives: np.ndarray):
        """InfoNCE in both directions; returns the loss and gradients for every layer."""
        n = len(anchors)
        activations, unit, norm = self._forward(np.vstack([anchors, positives]))
        a, p = unit[:n], unit[n:]
        logits = a @ p.T / self.temperature

        def softmax(m):
            m = m - m.max(axis=1, keepdims=True)
            e = np.exp(m)
            return e / e.sum(axis=1, keepdims=True)

        rows, cols = softmax(logits), softmax(logits.T).T
        eye = np.eye(n)
        diag = np.arange(n)
        loss = -0.5 * float(np.mean(np.log(rows[diag, diag] + 1e-12) + np.log(cols[diag, diag] + 1e-12)))
        d_logits = 0.5 * ((rows - eye) + (cols - eye)) / n
        d_unit = np.vstack([d_logits @ p, d_logits.T @ a]) / self.temperature
        # through the L2 normalisation
        delta = (d_unit - unit * np.sum(unit * d_unit, axis=1, keepdims=True)) / norm
        grads_w, grads_b = [None] * len(self.weights), [None] * len(self.weights)
        for layer in reversed(range(len(self.weights))):
            if layer < len(self.weights) - 1:
                delta = delta * (activations[layer + 1] > 0)
            grads_w[layer] = activations[layer].T @ delta
            grads_b[layer] = delta.sum(axis=0)
            if layer > 0:
                delta = delta @ self.weights[layer].T
        return loss, grads_w, grads_b

    def _augment(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = x + self.noise * rng.normal(size=x.shape)
        if self.feature_drop > 0:
            x = x * (rng.random(x.shape) >= self.feature_drop)
        return x

    # ------------------------------------------------------------------ training
    def fit(self, X, player_ids, X_val=None, val_ids=None) -> "ContrastiveEncoder":
        """Train on players with two or more seasons; early-stop on validation players."""
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        self._init(X.shape[1], rng)
        groups = list(_pairs_by_player(player_ids).values())
        val_groups = list(_pairs_by_player(val_ids).values()) if X_val is not None else []
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
        best, best_params, stale = np.inf, None, 0
        self.history_ = []

        for _ in range(self.epochs):
            order = rng.permutation(len(groups))
            running, batches = 0.0, 0
            for start in range(0, len(order) - 1, self.batch_size):
                chosen = [rng.choice(groups[g], size=2, replace=False) for g in order[start:start + self.batch_size]]
                if len(chosen) < 2:
                    continue
                first = self._augment(X[[c[0] for c in chosen]], rng)
                second = self._augment(X[[c[1] for c in chosen]], rng)
                loss, grads_w, grads_b = self._loss_and_grads(first, second)
                running, batches, step = running + loss, batches + 1, step + 1
                scale = self.lr * np.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                for i in range(len(self.weights)):
                    g = grads_w[i] + self.weight_decay * self.weights[i]
                    m_w[i] = beta1 * m_w[i] + (1 - beta1) * g
                    v_w[i] = beta2 * v_w[i] + (1 - beta2) * g * g
                    self.weights[i] -= scale * m_w[i] / (np.sqrt(v_w[i]) + eps)
                    m_b[i] = beta1 * m_b[i] + (1 - beta1) * grads_b[i]
                    v_b[i] = beta2 * v_b[i] + (1 - beta2) * grads_b[i] ** 2
                    self.biases[i] -= scale * m_b[i] / (np.sqrt(v_b[i]) + eps)
            if val_groups:
                first = np.asarray(X_val, float)[[g[0] for g in val_groups]]
                second = np.asarray(X_val, float)[[g[1] for g in val_groups]]
                val_loss = self._loss_and_grads(first, second)[0]
            else:
                val_loss = running / max(batches, 1)
            self.history_.append((running / max(batches, 1), val_loss))
            if val_loss < best - 1e-4:
                best, stale = val_loss, 0
                best_params = ([w.copy() for w in self.weights], [b.copy() for b in self.biases])
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.weights, self.biases = best_params
        self.best_val_loss_ = best
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(X, float))[1]

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez_compressed(path, temperature=self.temperature, **arrays)

    @classmethod
    def load(cls, path) -> "ContrastiveEncoder":
        data = np.load(path)
        layers = sum(1 for k in data.files if k.startswith("w"))
        model = cls(temperature=float(data["temperature"]))
        model.weights = [data[f"w{i}"] for i in range(layers)]
        model.biases = [data[f"b{i}"] for i in range(layers)]
        model.hidden = tuple(w.shape[1] for w in model.weights[:-1])
        model.dim = model.weights[-1].shape[1]
        return model
