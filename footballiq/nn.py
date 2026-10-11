"""Masked multi-view autoencoder in plain NumPy.

The model is small (two dense layers each side of a linear bottleneck), so it is
implemented directly with NumPy: it trains in a few seconds on a laptop CPU, has no
deep-learning dependency, and is bit-for-bit reproducible from the seed.

Training objective
------------------
Each player is described by several *views* (here: on-pitch performance and scouted
attributes). During training a random subset of the inputs is hidden:

* with probability ``view_drop`` one whole view is removed, and
* every remaining feature is hidden independently with probability ``feature_mask``.

The network sees the surviving values plus a 0/1 indicator of what is visible, and
must reconstruct *all* features of *all* views. To do that it has to learn how the
views relate to each other, which gives two properties the plain autoencoder lacks:

* the bottleneck cannot simply copy its input (the original 63 -> 64 autoencoder could);
* a player with only one view available (for example an EA record but no FBref
  statistics) can still be embedded in the same space.
"""
from __future__ import annotations

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class MaskedMultiViewAutoencoder:
    """Autoencoder trained with feature masking and whole-view dropout.

    Parameters
    ----------
    view_dims : sizes of the views, in the column order of the input matrix.
    latent_dim : width of the linear bottleneck (the embedding).
    hidden : hidden-layer widths of the encoder (mirrored in the decoder).
    feature_mask, view_drop : corruption rates described in the module docstring.
        Setting both to 0 gives an ordinary bottleneck autoencoder.
    latent_activation : ``"linear"`` (default) or ``"relu"`` (as in the legacy model).
    dropout : dropout rate on encoder hidden layers (legacy model used 0.2).
    """

    def __init__(self, view_dims, latent_dim=16, hidden=(128,), feature_mask=0.25,
                 view_drop=0.3, latent_activation="linear", dropout=0.0,
                 lr=2e-3, weight_decay=1e-4, batch_size=64, epochs=400, patience=30,
                 val_fraction=0.1, seed=0):
        self.view_dims = list(view_dims)
        self.latent_dim = latent_dim
        self.hidden = tuple(hidden)
        self.feature_mask = feature_mask
        self.view_drop = view_drop
        self.latent_activation = latent_activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed
        self.n_features = int(sum(self.view_dims))
        bounds = np.cumsum([0] + self.view_dims)
        self.view_slices = [slice(bounds[i], bounds[i + 1]) for i in range(len(self.view_dims))]
        # every view contributes equally to the loss, whatever its width
        self.loss_weights = np.concatenate(
            [np.full(d, 1.0 / (len(self.view_dims) * d)) for d in self.view_dims]
        )
        self.history_: list[tuple[float, float]] = []

    # ------------------------------------------------------------------ parameters
    def _init_params(self, rng: np.random.Generator) -> None:
        sizes = [2 * self.n_features, *self.hidden, self.latent_dim,
                 *reversed(self.hidden), self.n_features]
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.weights.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))
        self.latent_layer = len(self.hidden)          # index of the layer producing the code

    def _activation(self, layer: int) -> str:
        if layer == len(self.weights) - 1:
            return "linear"
        if layer == self.latent_layer:
            return self.latent_activation
        return "relu"

    # ------------------------------------------------------------------ forward / backward
    def _forward(self, x, visible, rng=None, upto=None):
        """Return the activations of every layer (``activations[0]`` is the input)."""
        weights, biases = self.weights, self.biases
        h = np.hstack([x * visible, visible])
        activations, drop_masks = [h], []
        last = len(weights) if upto is None else upto + 1
        for layer in range(last):
            z = h @ weights[layer] + biases[layer]
            h = _relu(z) if self._activation(layer) == "relu" else z
            keep = None
            if rng is not None and self.dropout > 0 and layer < self.latent_layer:
                keep = (rng.random(h.shape) >= self.dropout) / (1.0 - self.dropout)
                h = h * keep
            drop_masks.append(keep)
            activations.append(h)
        return activations, drop_masks

    def _loss_and_grads(self, x, visible, rng=None):
        activations, drop_masks = self._forward(x, visible, rng)
        error = activations[-1] - x
        loss = float(np.mean(np.sum(self.loss_weights * error ** 2, axis=1)))
        delta = 2.0 * error * self.loss_weights / len(x)
        grads_w, grads_b = [None] * len(self.weights), [None] * len(self.weights)
        for layer in reversed(range(len(self.weights))):
            out = activations[layer + 1]
            if drop_masks[layer] is not None:
                delta = delta * drop_masks[layer]
            if self._activation(layer) == "relu":
                delta = delta * (out > 0)
            grads_w[layer] = activations[layer].T @ delta
            grads_b[layer] = delta.sum(axis=0)
            if layer > 0:
                delta = delta @ self.weights[layer].T
        return loss, grads_w, grads_b

    # ------------------------------------------------------------------ corruption
    def _sample_visibility(self, n: int, rng: np.random.Generator) -> np.ndarray:
        visible = rng.random((n, self.n_features)) >= self.feature_mask
        if self.view_drop > 0 and len(self.view_dims) > 1:
            drop = rng.random(n) < self.view_drop
            which = rng.integers(0, len(self.view_dims), size=n)
            for v, sl in enumerate(self.view_slices):
                visible[drop & (which == v), sl] = False
        return visible.astype(float)

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray) -> "MaskedMultiViewAutoencoder":
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        self._init_params(rng)
        order = rng.permutation(len(X))
        n_val = max(1, int(round(self.val_fraction * len(X))))
        val, train = X[order[:n_val]], X[order[n_val:]]
        val_visible = self._sample_visibility(len(val), np.random.default_rng(self.seed + 1))

        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
        best, best_params, stale = np.inf, None, 0
        self.history_ = []

        for _ in range(self.epochs):
            perm = rng.permutation(len(train))
            running, batches = 0.0, 0
            for start in range(0, len(train), self.batch_size):
                batch = train[perm[start:start + self.batch_size]]
                visible = self._sample_visibility(len(batch), rng)
                loss, grads_w, grads_b = self._loss_and_grads(batch, visible, rng)
                running, batches, step = running + loss, batches + 1, step + 1
                scale = self.lr * np.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                for i in range(len(self.weights)):
                    g = grads_w[i] + self.weight_decay * self.weights[i]
                    m_w[i] = beta1 * m_w[i] + (1 - beta1) * g
                    v_w[i] = beta2 * v_w[i] + (1 - beta2) * g * g
                    self.weights[i] -= scale * m_w[i] / (np.sqrt(v_w[i]) + eps)
                    g = grads_b[i]
                    m_b[i] = beta1 * m_b[i] + (1 - beta1) * g
                    v_b[i] = beta2 * v_b[i] + (1 - beta2) * g * g
                    self.biases[i] -= scale * m_b[i] / (np.sqrt(v_b[i]) + eps)
            val_loss = self._loss_and_grads(val, val_visible)[0]
            self.history_.append((running / batches, val_loss))
            if val_loss < best - 1e-5:
                best, stale = val_loss, 0
                best_params = ([w.copy() for w in self.weights], [b.copy() for b in self.biases])
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.weights, self.biases = best_params
        self.best_val_loss_ = best
        return self

    # ------------------------------------------------------------------ inference
    def _visibility(self, X, available):
        if available is None:
            return np.ones_like(X, dtype=float)
        return np.asarray(available, float)

    def transform(self, X: np.ndarray, available: np.ndarray | None = None) -> np.ndarray:
        """Embed players. ``available`` marks which inputs are known (default: all)."""
        X = np.nan_to_num(np.asarray(X, float))
        return self._forward(X, self._visibility(X, available), upto=self.latent_layer)[0][-1]

    def reconstruct(self, X: np.ndarray, available: np.ndarray | None = None) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, float))
        return self._forward(X, self._visibility(X, available))[0][-1]

    def view_mask(self, n: int, views) -> np.ndarray:
        """Availability mask with only the listed view indices visible."""
        mask = np.zeros((n, self.n_features), dtype=bool)
        for v in views:
            mask[:, self.view_slices[v]] = True
        return mask

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        config = dict(view_dims=self.view_dims, latent_dim=self.latent_dim, hidden=self.hidden,
                      feature_mask=self.feature_mask, view_drop=self.view_drop,
                      latent_activation=self.latent_activation, dropout=self.dropout,
                      seed=self.seed)
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez_compressed(path, config=np.array(repr(config)), **arrays)

    @classmethod
    def load(cls, path) -> "MaskedMultiViewAutoencoder":
        import ast
        data = np.load(path, allow_pickle=False)
        model = cls(**ast.literal_eval(str(data["config"])))
        n_layers = sum(1 for k in data.files if k.startswith("w"))
        model.weights = [data[f"w{i}"] for i in range(n_layers)]
        model.biases = [data[f"b{i}"] for i in range(n_layers)]
        model.latent_layer = len(model.hidden)
        return model
