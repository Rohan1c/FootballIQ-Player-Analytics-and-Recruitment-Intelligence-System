"""Player embeddings from the two feature views.

Two interchangeable back-ends share one interface:

* ``"pca"``  - principal components of the standardised two-view features, with
               ridge regression filling in a view that is missing. This is the
               default: in the benchmark it matches or beats the autoencoders on
               every metric and is deterministic.
* ``"mdae"`` - the masked multi-view autoencoder from :mod:`footballiq.nn`.

Both can embed a player for whom only one view is known, which is what lets the
scouting module place players from leagues without FBref statistics in the same space.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

from .nn import MaskedMultiViewAutoencoder

RIDGE_ALPHAS = np.logspace(-1, 3, 9)


class Embedder:
    def __init__(self, method: str = "pca", dim: int = 16, seed: int = 0, **model_kwargs):
        if method not in ("pca", "mdae"):
            raise ValueError(f"unknown embedding method: {method}")
        self.method, self.dim, self.seed, self.model_kwargs = method, dim, seed, model_kwargs

    def fit(self, performance: np.ndarray, attributes: np.ndarray) -> "Embedder":
        self.n_perf, self.n_attr = performance.shape[1], attributes.shape[1]
        X = np.hstack([performance, attributes])
        if self.method == "pca":
            self.pca = PCA(self.dim, random_state=self.seed).fit(X)
            # cross-view imputers: predict the missing view from the observed one
            self.perf_from_attr = RidgeCV(alphas=RIDGE_ALPHAS).fit(attributes, performance)
            self.attr_from_perf = RidgeCV(alphas=RIDGE_ALPHAS).fit(performance, attributes)
        else:
            self.model = MaskedMultiViewAutoencoder(
                [self.n_perf, self.n_attr], latent_dim=self.dim, seed=self.seed, **self.model_kwargs
            ).fit(X)
        return self

    def transform(self, performance=None, attributes=None) -> np.ndarray:
        """Embed players; pass ``None`` for a view that is not available."""
        if performance is None and attributes is None:
            raise ValueError("at least one view is required")
        if self.method == "pca":
            if performance is None:
                performance = self.perf_from_attr.predict(attributes)
            if attributes is None:
                attributes = self.attr_from_perf.predict(performance)
            return self.pca.transform(np.hstack([performance, attributes]))
        n = len(performance if performance is not None else attributes)
        views = [v for v, given in ((0, performance), (1, attributes)) if given is not None]
        X = np.hstack([
            performance if performance is not None else np.zeros((n, self.n_perf)),
            attributes if attributes is not None else np.zeros((n, self.n_attr)),
        ])
        return self.model.transform(X, self.model.view_mask(n, views))
