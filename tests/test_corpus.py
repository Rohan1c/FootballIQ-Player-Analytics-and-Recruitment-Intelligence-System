"""Tests for the multi-season corpus, the identity metrics and the historical explorer."""
import unittest

import numpy as np

from footballiq import corpus, history
from footballiq.metric import ContrastiveEncoder, LinearIdentityMetric


def synthetic_players(n_players=300, n_seasons=3, signal=6, noise=10, seed=0):
    """Players have a stable signature in a few dimensions and large noise in the others."""
    rng = np.random.default_rng(seed)
    identity = rng.normal(size=(n_players, signal))
    rows, ids = [], []
    for p in range(n_players):
        for _ in range(n_seasons):
            rows.append(np.r_[identity[p] + 0.3 * rng.normal(size=signal), 2.0 * rng.normal(size=noise)])
            ids.append(p)
    return np.array(rows), np.array(ids)


def top1(Z, ids):
    dist = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(np.mean(ids[dist.argmin(axis=1)] == ids))


class Corpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seasons = corpus.load_seasons()
        cls.cohort = corpus.outfield(cls.seasons)

    def test_shape_and_keys(self):
        self.assertEqual(sorted(self.seasons["season"].unique()), [2018, 2019, 2020, 2021, 2022])
        self.assertFalse(self.seasons.duplicated(["season", "player_id"]).any())
        self.assertGreater(len(self.cohort), 7000)
        self.assertFalse((self.cohort["tm_position"] == "Goalkeeper").any())

    def test_known_player_season(self):
        kdb = self.seasons[(self.seasons["player"] == "Kevin De Bruyne") & (self.seasons["season"] == 2020)].iloc[0]
        self.assertEqual(kdb["squad"], "Manchester City")
        self.assertEqual(kdb["tm_position"], "Attacking Midfield")
        self.assertGreater(kdb["KeyPasses"], 100)

    def test_club_spells_are_summed_within_a_season(self):
        moved = self.seasons[self.seasons["n_spells"] > 1].iloc[0]
        self.assertIn(" / ", moved["squads"])

    def test_features_are_finite_and_thin_is_a_subset(self):
        rich = corpus.feature_table(self.cohort)
        thin = corpus.feature_table(self.cohort, thin=True)
        self.assertTrue(np.isfinite(rich.to_numpy()).all())
        self.assertEqual(thin.shape[1], 13)
        self.assertGreater(rich.shape[1], 80)
        for share in ("PassCompletion", "TouchShareAtt3rd", "DribbleSuccess"):
            self.assertTrue(rich[share].between(0, 1).all())

    def test_transfers_reference_corpus_players(self):
        transfers = corpus.load_transfers()
        self.assertGreater(transfers["player_id"].isin(self.seasons["player_id"]).mean(), 0.5)
        self.assertEqual(set(transfers["transfer_type"]), {"Arrivals", "Departures"})


class IdentityMetrics(unittest.TestCase):
    def test_linear_metric_downweights_unstable_directions(self):
        X, ids = synthetic_players()
        train, test = ids < 200, ids >= 200
        model = LinearIdentityMetric(dim=6).fit(X[train], ids[train])
        self.assertGreater(top1(model.transform(X[test]), ids[test]), top1(X[test], ids[test]) + 0.2)

    def test_contrastive_gradients(self):
        rng = np.random.default_rng(0)
        model = ContrastiveEncoder(dim=4, hidden=(6,))
        model._init(5, rng)
        a, p = rng.normal(size=(7, 5)), rng.normal(size=(7, 5))
        _, grads, _ = model._loss_and_grads(a, p)
        eps = 1e-6
        for layer, weights in enumerate(model.weights):
            for _ in range(5):
                r, c = rng.integers(weights.shape[0]), rng.integers(weights.shape[1])
                weights[r, c] += eps
                up = model._loss_and_grads(a, p)[0]
                weights[r, c] -= 2 * eps
                down = model._loss_and_grads(a, p)[0]
                weights[r, c] += eps
                self.assertAlmostEqual((up - down) / (2 * eps), grads[layer][r, c], places=6)

    def test_contrastive_encoder_learns_identity(self):
        X, ids = synthetic_players()
        train, val, test = ids < 160, (ids >= 160) & (ids < 200), ids >= 200
        model = ContrastiveEncoder(dim=6, hidden=(), temperature=0.05, feature_drop=0.0, lr=1e-2, seed=0)
        model.fit(X[train], ids[train], X[val], ids[val])
        self.assertGreater(top1(model.transform(X[test]), ids[test]), top1(X[test], ids[test]) + 0.2)
        again = ContrastiveEncoder(dim=6, hidden=(), temperature=0.05, feature_drop=0.0, lr=1e-2, seed=0)
        again.fit(X[train], ids[train], X[val], ids[val])
        np.testing.assert_array_equal(model.transform(X[test]), again.transform(X[test]))


class HistoricalExplorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = history.get_history()

    def test_find_by_season(self):
        row = self.store.find("Kevin De Bruyne", 2020)
        self.assertEqual(self.store.table.at[row, "season"], 2020)
        self.assertEqual(self.store.table.at[self.store.find("Kevin De Bruyne"), "season"], 2022)
        self.assertIsNone(self.store.find("Kevin De Bruyne", 2030))
        self.assertIsNone(self.store.find("zzzzzz"))

    def test_similar_seasons_respects_filters(self):
        results = history.similar_seasons("Mohamed Salah", 2018, 8, same_season=True, max_value_eur=20e6)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r["season"] == 2018 for r in results))
        self.assertTrue(all(r["market_value_eur"] <= 20e6 for r in results))
        self.assertNotIn("Mohamed Salah", [r["player"] for r in results])
        self.assertEqual(len({r["player"] for r in results}), 8)          # one season per player
        scores = [r["similarity"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_a_player_resembles_himself_next_season(self):
        timeline = history.style_timeline("Joshua Kimmich")
        self.assertEqual([e["season"] for e in timeline], [2018, 2019, 2020, 2021, 2022])
        self.assertIsNone(timeline[0]["change"])
        self.assertTrue(all(e["change"] < 25 for e in timeline[1:]))


if __name__ == "__main__":
    unittest.main()
