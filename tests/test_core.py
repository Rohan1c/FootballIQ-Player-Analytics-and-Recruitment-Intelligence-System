"""Unit tests for the data layer and the model. Run:  python -m unittest discover -s tests"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from footballiq import features, linkage
from footballiq.nn import MaskedMultiViewAutoencoder


class NameMatching(unittest.TestCase):
    def test_normalise_strips_accents_and_punctuation(self):
        self.assertEqual(linkage.normalise_name("Martin Ødegaard"), "martin odegaard")
        self.assertEqual(linkage.normalise_name("Obite N'Dicka"), "obite ndicka")
        self.assertEqual(linkage.normalise_name("Warren Zaïre-Emery"), "warren zaire emery")

    def test_similarity_rules(self):
        sim = linkage.name_similarity
        self.assertEqual(sim("pedri", "pedri"), 1.0)
        self.assertGreaterEqual(sim("gabriel magalhaes", "gabriel"), 0.9)       # token containment
        self.assertGreaterEqual(sim("vinicius junior", "vini jr"), 0.85)        # abbreviation
        self.assertLess(sim("joshua kimmich", "leon goretzka"), 0.6)


class Linkage(unittest.TestCase):
    def setUp(self):
        base = {"Nation": "es ESP", "Comp": "es La Liga", "Born": 2000, "Age": 25, "Pos": "MF"}
        zero = {c: 0 for c in linkage.COUNT_COLS}
        rows = [
            {**base, **zero, "Player": "Ana Pérez", "Squad": "Alpha", "Min": 900, "Gls": 3},
            {**base, **zero, "Player": "Ana Pérez", "Squad": "Beta", "Min": 300, "Gls": 1},   # second spell
            {**base, **zero, "Player": "Luis Gómez", "Squad": "Alpha", "Min": 500},
            {**base, **zero, "Player": "Solo", "Squad": "Beta", "Min": 400},                 # nickname
            {**base, **zero, "Player": "Old Keeper", "Squad": "Alpha", "Min": 900, "Pos": "GK"},
        ]
        # filler players so the club and nation spellings can be learnt
        for k in range(6):
            rows.append({**base, **zero, "Player": f"Filler Alpha{chr(97 + k)}", "Squad": "Alpha", "Min": 100})
            rows.append({**base, **zero, "Player": f"Filler Beta{chr(97 + k)}", "Squad": "Beta", "Min": 100})
        self.fbref = pd.DataFrame(rows)
        attrs = {c: 60 for c in features.ATTRIBUTE_FEATURES + features.FACE_RATINGS}
        ea = [
            {"Name": "Ana Perez", "Team": "Alpha FC", "Age": 24, "Position": "CM"},
            {"Name": "Luis Gomez", "Team": "Alpha FC", "Age": 40, "Position": "CM"},        # wrong person: age
            {"Name": "Solo", "Team": "Gamma", "Age": 25, "Position": "ST"},                   # wrong club
            {"Name": "Old Keeper", "Team": "Alpha FC", "Age": 25, "Position": "CB"},          # GK vs outfield
        ]
        for k in range(6):
            ea.append({"Name": f"Filler Alpha{chr(97 + k)}", "Team": "Alpha FC", "Age": 25, "Position": "CM"})
            ea.append({"Name": f"Filler Beta{chr(97 + k)}", "Team": "Beta CF", "Age": 25, "Position": "CM"})
        self.ea = pd.DataFrame([{**attrs, "Nation": "Spain", "League": "L", "player_id": i,
                                 "Preferred Foot": "Right", **row} for i, row in enumerate(ea)])

    def test_club_spells_are_summed(self):
        agg = linkage.aggregate_fbref(self.fbref)
        ana = agg[agg["Player"] == "Ana Pérez"].iloc[0]
        self.assertEqual((ana["Min"], ana["Gls"], ana["n_spells"]), (1200, 4, 2))
        self.assertEqual(ana["Squad"], "Alpha")                                   # club with most minutes

    def test_matches_and_rejections(self):
        linked = linkage.link(self.fbref, self.ea).set_index("Player")
        self.assertIn("Ana Pérez", linked.index)                   # accent-insensitive
        self.assertEqual(linked.loc["Ana Pérez", "Min"], 1200)
        self.assertNotIn("Luis Gómez", linked.index)               # ages 15 years apart
        self.assertNotIn("Solo", linked.index)                     # one-word name needs the club
        self.assertNotIn("Old Keeper", linked.index)               # goalkeeper cannot match outfield


class Shrinkage(unittest.TestCase):
    def test_rate_is_pulled_towards_group_mean_when_minutes_are_low(self):
        rng = np.random.default_rng(0)
        exposure = np.r_[rng.uniform(10, 38, 200), 1.0, 30.0]
        true_rate = rng.gamma(4, 0.05, size=202)
        count = rng.poisson(true_rate * exposure).astype(float)
        count[-2], count[-1] = 1.0, 30.0                 # both score 1.0 per 90 on paper
        est = features.shrink_rate(count, exposure, np.full(202, "FW", dtype=object))
        mean = count[:200].sum() / exposure[:200].sum()
        self.assertLess(est[-2], est[-1])                # 1 match is trusted less than 30
        self.assertLess(abs(est[-2] - mean), abs(1.0 - mean))
        self.assertTrue(np.all(est >= 0))

    def test_ratio_handles_zero_trials(self):
        est = features.shrink_ratio([0, 5, 30], [0, 10, 60], np.full(3, "FW", dtype=object))
        self.assertTrue(np.all(np.isfinite(est)))
        self.assertTrue(np.all((est > 0) & (est < 1)))


class Autoencoder(unittest.TestCase):
    def test_backpropagation_matches_numerical_gradient(self):
        rng = np.random.default_rng(0)
        model = MaskedMultiViewAutoencoder([3, 4], latent_dim=2, hidden=(5,), seed=1)
        model._init_params(rng)
        x = rng.normal(size=(6, 7))
        visible = model._sample_visibility(6, rng)
        _, grads, _ = model._loss_and_grads(x, visible)
        eps = 1e-6
        for layer, weights in enumerate(model.weights):
            for _ in range(5):
                a, b = rng.integers(weights.shape[0]), rng.integers(weights.shape[1])
                weights[a, b] += eps
                up = model._loss_and_grads(x, visible)[0]
                weights[a, b] -= 2 * eps
                down = model._loss_and_grads(x, visible)[0]
                weights[a, b] += eps
                self.assertAlmostEqual((up - down) / (2 * eps), grads[layer][a, b], places=6)

    def test_training_is_reproducible_and_survives_save_load(self):
        rng = np.random.default_rng(0)
        latent = rng.normal(size=(300, 3))
        X = np.hstack([latent @ rng.normal(size=(3, 5)), latent @ rng.normal(size=(3, 6))])
        X += 0.1 * rng.normal(size=X.shape)
        first = MaskedMultiViewAutoencoder([5, 6], latent_dim=3, epochs=60, seed=3).fit(X)
        second = MaskedMultiViewAutoencoder([5, 6], latent_dim=3, epochs=60, seed=3).fit(X)
        np.testing.assert_array_equal(first.transform(X), second.transform(X))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            first.save(path)
            reloaded = MaskedMultiViewAutoencoder.load(path)
        np.testing.assert_allclose(first.transform(X), reloaded.transform(X))

    def test_missing_view_is_predicted_from_the_other(self):
        rng = np.random.default_rng(1)
        latent = rng.normal(size=(600, 3))
        X = np.hstack([latent @ rng.normal(size=(3, 5)), latent @ rng.normal(size=(3, 6))])
        X = (X - X.mean(0)) / X.std(0)
        model = MaskedMultiViewAutoencoder([5, 6], latent_dim=3, epochs=150, seed=0).fit(X[:500])
        test = X[500:]
        recon = model.reconstruct(test, model.view_mask(len(test), [1]))     # view 0 hidden
        error = ((recon[:, :5] - test[:, :5]) ** 2).mean()
        self.assertLess(error, 0.5)                                          # 1.0 = predicting the mean


if __name__ == "__main__":
    unittest.main()
