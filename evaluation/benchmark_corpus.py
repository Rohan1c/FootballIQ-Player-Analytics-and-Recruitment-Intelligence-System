"""Benchmark on the multi-season corpus (big five leagues, 2017-18 to 2021-22).

Everything is evaluated on players the models have never seen: players are split
60 / 15 / 25 into train, validation and test, and the split is repeated five times.

Tables written to ``evaluation/results/``:

  corpus_cohort.csv            size of the corpus and of the evaluation cohort
  corpus_representations.csv   every representation on the same held-out players
  corpus_reid_by_move.csv      re-identification for players who stayed / changed club
  corpus_roles.csv             role classification with thin and rich statistics
  corpus_transfers.csv         do clubs sign players who resemble the ones who left?

Run from the repository root:  python -m evaluation.benchmark_corpus
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import common, metrics
from footballiq import config, corpus
from footballiq.linkage import normalise_name
from footballiq.metric import ContrastiveEncoder, LinearIdentityMetric
from footballiq.nn import MaskedMultiViewAutoencoder
from footballiq.roles import POSITION_ROLE_MAP, RoleClassifier

warnings.filterwarnings("ignore")
SPLITS = [0, 1, 2, 3, 4]
DIM = 32


def split_players(player_ids: np.ndarray, seed: int):
    """Boolean row masks for train / validation / test, split by player."""
    players = np.unique(player_ids)
    order = np.random.default_rng(seed).permutation(players)
    n = len(players)
    train, val = set(order[: int(0.60 * n)]), set(order[int(0.60 * n): int(0.75 * n)])
    is_train = np.array([p in train for p in player_ids])
    is_val = np.array([p in val for p in player_ids])
    return is_train, is_val, ~(is_train | is_val)


class Cohort:
    """Qualified outfield player-seasons with both feature sets and the lookup tables."""

    def __init__(self):
        self.table = corpus.outfield(corpus.load_seasons())
        self.rich = corpus.feature_table(self.table).to_numpy()
        self.thin = corpus.feature_table(self.table, thin=True).to_numpy()
        self.ids = self.table["player_id"].to_numpy(object)
        self.season = self.table["season"].to_numpy()
        self.squad = self.table["squad"].to_numpy(object)
        self.position = self.table["tm_position"].to_numpy(object)
        self.family = self.table["tm_position"].map(corpus.POSITION_FAMILY).to_numpy(object)
        self.log_minutes = np.log(self.table["Min"].to_numpy(float))
        self.row = {(p, s): i for i, (p, s) in enumerate(zip(self.ids, self.season))}
        # consecutive-season pairs: (row in season t, row in season t + 1)
        self.pairs = np.array([(i, self.row[(p, s + 1)]) for i, (p, s) in enumerate(zip(self.ids, self.season))
                               if (p, s + 1) in self.row])
        self.moved = self.squad[self.pairs[:, 0]] != self.squad[self.pairs[:, 1]]


def reid_ranks(Z: np.ndarray, cohort: Cohort, query_mask: np.ndarray) -> np.ndarray:
    """Rank of a player's next season among every player-season of that next season.

    Returns one rank per consecutive-season pair whose first season is in ``query_mask``
    (NaN for the other pairs), aligned with ``cohort.pairs``.
    """
    ranks = np.full(len(cohort.pairs), np.nan)
    for season in np.unique(cohort.season):
        gallery = np.flatnonzero(cohort.season == season + 1)
        chosen = np.flatnonzero((cohort.season[cohort.pairs[:, 0]] == season) & query_mask[cohort.pairs[:, 0]])
        if len(gallery) == 0 or len(chosen) == 0:
            continue
        queries, targets = cohort.pairs[chosen, 0], cohort.pairs[chosen, 1]
        dist = np.linalg.norm(Z[queries][:, None, :] - Z[gallery][None, :, :], axis=2)
        true = np.linalg.norm(Z[queries] - Z[targets], axis=1)
        ranks[chosen] = (dist < true[:, None] - 1e-12).sum(axis=1) + 1
    return ranks


def purity_other_players(Z, labels, ids, k: int = 10) -> float:
    """Neighbour purity where a player's own other seasons do not count as neighbours."""
    nb = NearestNeighbors(n_neighbors=min(k + 8, len(Z) - 1) + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    scores = []
    for i, row in enumerate(nb):
        other = [j for j in row if ids[j] != ids[i]][:k]
        scores.append(np.mean(labels[other] == labels[i]))
    return float(np.mean(scores))


def representations_for_split(cohort: Cohort, seed: int) -> dict:
    """Fit every representation on the training players of one split."""
    train, val, _ = split_players(cohort.ids, seed)
    rich = corpus.Standardiser.fit(cohort.rich[train]).transform(cohort.rich)
    thin = corpus.Standardiser.fit(cohort.thin[train]).transform(cohort.thin)
    ids = cohort.ids
    n_rich = rich.shape[1]
    out = {
        f"thin statistics ({thin.shape[1]}), standardised": thin,
        f"thin statistics + identity metric": LinearIdentityMetric(thin.shape[1]).fit(thin[train], ids[train]).transform(thin),
        f"rich statistics ({n_rich}), standardised": rich,
        f"rich + PCA-{DIM}": PCA(DIM, random_state=seed).fit(rich[train]).transform(rich),
        f"rich + autoencoder-{DIM}": MaskedMultiViewAutoencoder(
            [n_rich], latent_dim=DIM, feature_mask=0.0, view_drop=0.0, seed=seed).fit(rich[train]).transform(rich),
        f"rich + masked autoencoder-{DIM}": MaskedMultiViewAutoencoder(
            [n_rich], latent_dim=DIM, view_drop=0.0, seed=seed).fit(rich[train]).transform(rich),
        f"rich + identity metric, linear (LDA-{DIM})": LinearIdentityMetric(DIM).fit(rich[train], ids[train]).transform(rich),
        f"rich + contrastive encoder, linear ({DIM})": ContrastiveEncoder(
            dim=DIM, hidden=(), temperature=0.05, weight_decay=1e-2, feature_drop=0.0, seed=seed
        ).fit(rich[train], ids[train], rich[val], ids[val]).transform(rich),
        f"rich + contrastive encoder, 1 hidden layer ({DIM})": ContrastiveEncoder(
            dim=DIM, hidden=(128,), temperature=0.05, weight_decay=1e-2, feature_drop=0.0, seed=seed
        ).fit(rich[train], ids[train], rich[val], ids[val]).transform(rich),
        f"rich + contrastive encoder, 2 hidden layers ({DIM})": ContrastiveEncoder(
            dim=DIM, hidden=(256, 128), temperature=0.05, weight_decay=1e-2, feature_drop=0.0, seed=seed
        ).fit(rich[train], ids[train], rich[val], ids[val]).transform(rich),
    }
    return out


def score_split(cohort: Cohort, seed: int, reps: dict) -> tuple[list, dict]:
    train, _, test = split_players(cohort.ids, seed)
    rows, ranks_by_rep = [], {}
    for name, Z in reps.items():
        ranks = reid_ranks(Z, cohort, test)
        r = ranks[~np.isnan(ranks)]
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Z[train], cohort.family[train])
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
        pred = cross_val_predict(ridge, Z[test], cohort.log_minutes[test], cv=KFold(5, shuffle=True, random_state=seed))
        target = cohort.log_minutes[test]
        club_season = np.array([f"{a}|{b}" for a, b in zip(cohort.squad[test], cohort.season[test])], dtype=object)
        rows.append({
            "representation": name, "dim": Z.shape[1], "split": seed, "reid_queries": len(r),
            "reid_top1": float(np.mean(r == 1)), "reid_top10": float(np.mean(r <= 10)),
            "reid_mrr": float(np.mean(1.0 / r)),
            "position_purity@10": purity_other_players(Z[test], cohort.position[test], cohort.ids[test]),
            "family_purity@10": purity_other_players(Z[test], cohort.family[test], cohort.ids[test]),
            "family_probe_F1": float(f1_score(cohort.family[test], probe.predict(Z[test]), average="macro")),
            "minutes_R2": float(1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum()),
            "same_club@10": purity_other_players(Z[test], club_season, cohort.ids[test]),
        })
        ranks_by_rep[name] = ranks
    return rows, ranks_by_rep


# ---------------------------------------------------------------------------- roles on the corpus
def roles_on_corpus(cohort: Cohort) -> pd.DataFrame:
    """Role classification for labelled players, using their latest qualified corpus season.

    The role labels describe the players as of 2025-26, so applying them to a season up to
    four years earlier is an approximation: a player's role can change. The comparison between
    feature sets is still like for like, because it uses the same players and the same folds.
    """
    labels = pd.concat([pd.read_csv(config.ROLE_LABELS), pd.read_csv(config.ROLE_LABELS_DRAFT)], ignore_index=True)
    labels = labels[labels["position"] != "GK"]
    table = cohort.table
    key = {}
    for i, (name, born, season) in enumerate(zip(table["player"], table["born"], table["season"])):
        k = (normalise_name(name), int(born) if born == born else -1)
        if k not in key or season > table.at[key[k], "season"]:
            key[k] = i
    rows, roles, positions = [], [], []
    for name, born, role, position in zip(labels["player"], labels["born"], labels["role"], labels["position"]):
        i = key.get((normalise_name(name), int(born)))
        if i is not None:
            rows.append(i); roles.append(role); positions.append(position)
    rows, roles, positions = np.array(rows), np.array(roles, dtype=object), np.array(positions, dtype=object)
    counts = pd.Series(roles).value_counts()
    strata = np.where(pd.Series(roles).map(counts).to_numpy() < 5, "rare", roles)

    rich = corpus.Standardiser.fit(cohort.rich).transform(cohort.rich)
    thin = corpus.Standardiser.fit(cohort.thin).transform(cohort.thin)
    feature_sets = {f"thin statistics ({thin.shape[1]})": thin[rows], f"rich statistics ({rich.shape[1]})": rich[rows]}
    out = []
    for name, X in feature_sets.items():
        for gate in (True, False):
            accs = []
            for rep in range(10):
                hits = []
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, strata):
                    model = RoleClassifier().fit(X[tr], roles[tr])
                    hits.extend(model.predict(X[te], positions[te] if gate else None) == roles[te])
                accs.append(np.mean(hits))
            out.append({"features": name, "position_gate": gate, "labelled_players": len(rows),
                        "accuracy": float(np.mean(accs)), "sd_over_repeats": float(np.std(accs))})
    chance = np.mean([(r in POSITION_ROLE_MAP[p]) / len(POSITION_ROLE_MAP[p]) for r, p in zip(roles, positions)])
    out.append({"features": "random choice within the position gate", "position_gate": True,
                "labelled_players": len(rows), "accuracy": float(chance), "sd_over_repeats": 0.0})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- transfers
def transfer_validation(cohort: Cohort, reps: dict) -> pd.DataFrame:
    """Do clubs sign players who resemble the ones they let go?

    For every club and summer where a qualified permanent signing and a qualified permanent
    departure play in the same position group, compare the signing with the departed player
    (both described by the season before the move). The reference set is every other club's
    permanent signing in that position group that summer. ``mean_percentile`` is the share of
    that market that is *closer* to the departed player than the club's actual signing: 0.5
    means clubs choose no differently from the market, lower means they pick look-alikes.
    """
    transfers = corpus.load_transfers()
    transfers = transfers[(transfers["is_loan"] == 0) & (transfers["window"] == "Summer")]
    transfers = transfers.assign(family=transfers["player_position"].map(corpus.POSITION_FAMILY))
    transfers = transfers.dropna(subset=["family"])
    # the season before the move describes the player
    transfers = transfers.assign(row=[cohort.row.get((p, s - 1)) for p, s in zip(transfers["player_id"], transfers["season"])])
    transfers = transfers.dropna(subset=["row"]).astype({"row": int})
    arrivals = transfers[transfers["transfer_type"] == "Arrivals"]
    departures = transfers[transfers["transfer_type"] == "Departures"]

    cases = []      # (departed row, signed row, market rows)
    for (season, family), market in arrivals.groupby(["season", "family"]):
        leaving = departures[(departures["season"] == season) & (departures["family"] == family)]
        for team, signed in market.groupby("team_name"):
            others = market.loc[market["team_name"] != team, "row"].to_numpy()
            out_rows = leaving.loc[leaving["team_name"] == team, "row"].to_numpy()
            if len(others) < 5:
                continue
            for d in out_rows:
                for a in signed["row"].to_numpy():
                    if a != d:
                        cases.append((d, a, others, f"{team}|{season}"))
    rng = np.random.default_rng(0)
    groups = np.array([c[3] for c in cases], dtype=object)
    unique_groups = np.unique(groups)
    rows = []
    for name, Z in reps.items():
        pct = np.array([np.mean(np.linalg.norm(Z[others] - Z[d], axis=1) < np.linalg.norm(Z[a] - Z[d]))
                        for d, a, others, _ in cases])
        by_group = pd.Series(pct).groupby(groups).mean()
        boot = [by_group.loc[rng.choice(unique_groups, len(unique_groups))].mean() for _ in range(5000)]
        rows.append({"representation": name, "club_summers": len(unique_groups), "signing_departure_pairs": len(cases),
                     "mean_percentile": float(by_group.mean()),
                     "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                     "share_in_closest_quarter": float(np.mean(pct <= 0.25))})
    return pd.DataFrame(rows)


def main() -> None:
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    cohort = Cohort()
    seasons = corpus.load_seasons()
    cohort_table = pd.DataFrame([
        {"item": "player-seasons in the corpus", "value": len(seasons)},
        {"item": "players in the corpus", "value": seasons["player_id"].nunique()},
        {"item": "qualified outfield player-seasons (900+ minutes)", "value": len(cohort.table)},
        {"item": "qualified outfield players", "value": len(np.unique(cohort.ids))},
        {"item": "consecutive-season pairs", "value": len(cohort.pairs)},
        {"item": "consecutive-season pairs with a change of club", "value": int(cohort.moved.sum())},
        {"item": "rich features", "value": cohort.rich.shape[1]},
        {"item": "thin features (what the 2025-26 FBref file still allows)", "value": cohort.thin.shape[1]},
    ])
    cohort_table.to_csv(common.RESULTS / "corpus_cohort.csv", index=False)

    all_rows, move_rows, first_reps = [], [], None
    for seed in SPLITS:
        reps = representations_for_split(cohort, seed)
        first_reps = first_reps or reps
        rows, ranks = score_split(cohort, seed, reps)
        all_rows.extend(rows)
        for name, r in ranks.items():
            valid = ~np.isnan(r)
            for label, mask in (("stayed at the same club", valid & ~cohort.moved), ("changed club", valid & cohort.moved)):
                move_rows.append({"representation": name, "group": label, "split": seed, "queries": int(mask.sum()),
                                  "reid_top1": float(np.mean(r[mask] == 1)), "reid_top10": float(np.mean(r[mask] <= 10))})
    per_split = pd.DataFrame(all_rows)
    metric_cols = [c for c in per_split.columns if c not in ("representation", "dim", "split")]
    mean = per_split.groupby("representation", sort=False)[metric_cols].mean()
    sd = per_split.groupby("representation", sort=False)[["reid_top1", "reid_top10", "family_purity@10"]].std()
    table = pd.concat([per_split.groupby("representation", sort=False)["dim"].first(), mean,
                       sd.add_suffix("_sd")], axis=1).reset_index()
    table.insert(2, "splits", len(SPLITS))
    table.to_csv(common.RESULTS / "corpus_representations.csv", index=False)

    moves = pd.DataFrame(move_rows).groupby(["representation", "group"], sort=False)[["queries", "reid_top1", "reid_top10"]].mean().reset_index()
    keep = [n for n in first_reps if "standardised" in n or "LDA" in n]
    moves = moves[moves["representation"].isin(keep)]
    moves.to_csv(common.RESULTS / "corpus_reid_by_move.csv", index=False)

    roles = roles_on_corpus(cohort)
    roles.to_csv(common.RESULTS / "corpus_roles.csv", index=False)

    transfer_reps = {n: first_reps[n] for n in first_reps if "standardised" in n or "LDA" in n or "PCA" in n}
    transfers = transfer_validation(cohort, transfer_reps)
    transfers.to_csv(common.RESULTS / "corpus_transfers.csv", index=False)

    for title, frame in [("corpus", cohort_table), ("representations (mean of 5 player splits)", table),
                         ("re-identification by change of club", moves), ("roles on the corpus", roles),
                         ("transfers", transfers)]:
        print(f"\n== {title} ==")
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 70)
    main()
