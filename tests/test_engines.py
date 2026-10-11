"""Behavioural tests of the engines on the real dataset.

Several of these are regression tests for bugs found in the original code.
"""
import time
import unittest

import numpy as np

from footballiq import roles
from footballiq.compatibility import best_partners, compatibility_row
from footballiq.scouting import global_lookalikes
from footballiq.similarity import similar_players
from footballiq.squad import XI_ORDER, best_trio, build_xi
from footballiq.store import get_store
from footballiq.transfer import find_replacements


class Engines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = get_store()

    def name(self, query):
        index = self.store.find(query)
        return None if index is None else self.store.names[index]

    def test_name_lookup_does_not_return_substring_accidents(self):
        # the original substring search returned Rodrigo Bentancur, Lorenzo Colombo, Brenden Aaronson
        self.assertEqual(self.name("Rodri"), "Rodri")
        self.assertEqual(self.name("Enzo"), "Enzo Fernández")
        self.assertIsNone(self.name("Son"))
        self.assertEqual(self.name("Odegaard"), "Martin Ødegaard")
        self.assertEqual(self.name("Vinicius"), "Vinicius Júnior")
        self.assertIsNone(self.name("zzzzzz"))

    def test_players_dropped_by_the_original_merge_are_present(self):
        for player in ["Vinicius Júnior", "Martín Zubimendi", "Gabriel Magalhães", "Mikel Oyarzabal"]:
            self.assertEqual(self.name(player), player)

    def test_club_spells_are_combined(self):
        rabiot = self.store.row(self.store.find("Adrien Rabiot"))
        self.assertGreater(rabiot["Min"], 2000)          # original kept only 90 minutes at Marseille
        self.assertEqual(rabiot["n_spells"], 2)

    def test_goalkeepers_and_outfield_players_never_mix(self):
        for query in ["Thibaut Courtois", "Pedri"]:
            is_gk = self.store.is_gk[self.store.find(query)]
            for r in similar_players(query, 10, same_position=False):
                self.assertEqual(self.store.is_gk[r["index"]], is_gk)

    def test_similarity_is_symmetric_bounded_and_self_is_one(self):
        i, j = self.store.find("Pedri"), self.store.find("Vitinha")
        self.assertAlmostEqual(self.store.similarity(i, i), 1.0)
        self.assertAlmostEqual(self.store.similarity(i, j), self.store.similarity(j, i), places=6)
        row = self.store.similarity_row(i)
        self.assertTrue(np.all((row >= 0) & (row <= 1)))

    def test_similar_players_are_sorted_and_explained(self):
        results = similar_players("Erling Haaland", 5)
        scores = [r["similarity"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(r["position"] == "ST" for r in results))
        self.assertTrue(results[0]["why"]["shared"])

    def test_roles_respect_the_position_gate(self):
        for i in range(self.store.n):
            if i in set(self.store.outfield[self.store.role_training_rows]):
                continue                                    # human labels are kept as given
            self.assertIn(self.store.roles[i], roles.POSITION_ROLE_MAP[self.store.positions[i]])

    def test_every_engine_reports_the_same_role(self):
        # originally the partner and replacement engines disagreed with the role engine for 35% of players
        i = self.store.find("Pedri")
        role = self.store.roles[i]
        self.assertEqual(max(self.store.role_scores(i), key=self.store.role_scores(i).get), role)
        partner = best_partners("Pedri", 1)[0]
        self.assertEqual(partner["role"], self.store.roles[partner["index"]])

    def test_role_probabilities_sum_to_one(self):
        np.testing.assert_allclose(self.store.role_proba.sum(axis=1), 1.0, atol=1e-9)

    def test_replacements_filters(self):
        target_age = self.store.ages[self.store.find("Robert Lewandowski")]
        results = find_replacements("Robert Lewandowski", 10, younger_only=True)
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r["age"] < target_age for r in results))
        self.assertEqual(find_replacements("zzzzzz"), [])

    def test_starting_xi_contains_the_anchor_in_his_position(self):
        # the original XI builder left out any anchor who was not a midfielder
        for query, slot in [("Erling Haaland", "ST"), ("Virgil van Dijk", "CB1"),
                            ("Bukayo Saka", "RW"), ("Pedri", "CM1"), ("Thibaut Courtois", "GK")]:
            xi = build_xi(query)
            self.assertEqual(sorted(xi), sorted(XI_ORDER))
            self.assertEqual(xi[slot], self.store.find(query))
            self.assertEqual(len(set(xi.values())), 11)
            self.assertTrue(self.store.is_gk[xi["GK"]])

    def test_midfield_trio(self):
        trio, score = best_trio("Pedri")
        self.assertEqual(len(set(trio)), 3)
        self.assertEqual(trio[0], self.store.find("Pedri"))
        self.assertGreater(score, 0)
        self.assertEqual(best_trio("Erling Haaland"), (None, 0))

    def test_global_scouting_excludes_linked_players_and_respects_filters(self):
        results = global_lookalikes("Pedri", 10, max_age=23)
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r["age"] <= 23 for r in results))
        linked = set(self.store.players["ea_name"])
        big_five = {"Premier League", "LALIGA EA SPORTS", "Bundesliga", "Serie A Enilive", "Ligue 1 McDonald's"}
        self.assertFalse(any(r["player"] in linked and r["league"] in big_five for r in results))

    def test_queries_are_fast(self):
        compatibility_row(0, self.store)                   # warm up
        start = time.perf_counter()
        best_partners("Pedri", 10)
        find_replacements("Pedri", 10)
        similar_players("Pedri", 10)
        self.assertLess(time.perf_counter() - start, 0.5)  # the original took about 23 s for these


if __name__ == "__main__":
    unittest.main()
