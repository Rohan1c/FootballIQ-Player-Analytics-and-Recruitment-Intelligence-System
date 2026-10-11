"""Benchmark of player representations.

Produces, under ``evaluation/results/``:

  data_linkage.csv        how many players each merge strategy keeps
  representations.csv     every representation scored on the same players
  dimension_sweep.csv     sensitivity to the embedding size
  missing_view.csv        embedding a player when one view is unavailable
  reidentification.csv    matching the two club spells of mid-season movers

Run from the repository root:  python -m evaluation.benchmark_embeddings
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from evaluation import common, metrics
from footballiq import config, features, linkage
from footballiq.embedding import RIDGE_ALPHAS
from footballiq.nn import MaskedMultiViewAutoencoder

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2, 3, 4]
DIM = config.EMBEDDING_DIM


def autoencoder(view_dims, dim, seed, masked):
    kwargs = {} if masked else {"feature_mask": 0.0, "view_drop": 0.0}
    return MaskedMultiViewAutoencoder(view_dims, latent_dim=dim, seed=seed, **kwargs)


# ---------------------------------------------------------------------------- 1. linkage
def data_linkage() -> pd.DataFrame:
    fbref = pd.read_csv(config.FBREF_RAW)
    aggregated = linkage.aggregate_fbref(fbref)
    legacy = pd.read_csv(config.LEGACY_MERGED)
    new = pd.read_csv(config.PLAYERS_CSV)
    total_minutes = fbref["Min"].sum()
    rows = [
        {"merge": "original (exact name, first club spell only)", "players": len(legacy),
         "share_of_fbref_players": len(legacy) / len(aggregated),
         "share_of_minutes": legacy["Min"].sum() / total_minutes,
         "age_gap_over_2y": int(((legacy["Age_x"] - legacy["Age_y"]).abs() > 2).sum())},
        {"merge": "record linkage (this work)", "players": len(new),
         "share_of_fbref_players": len(new) / len(aggregated),
         "share_of_minutes": new["Min"].sum() / total_minutes,
         "age_gap_over_2y": int(((new["Age"] - new["ea_age"]).abs() > 2).sum())},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- 2. representations
def representations() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, views, outfield = common.load_new()
    legacy, legacy_X, legacy_Z = common.load_legacy()
    cohort, legacy_rows = common.common_cohort(outfield, legacy)
    train = outfield["Min"].to_numpy() >= config.MIN_MINUTES
    X, P, A = views.combined, views.performance, views.attributes
    dims = [P.shape[1], A.shape[1]]

    position = outfield["Position"].to_numpy(object)[cohort]
    family = outfield["Position"].map(common.POSITION_FAMILY).to_numpy(object)[cohort]
    log_minutes = np.log(outfield["Min"].to_numpy(float)[cohort])
    club = outfield["Squad"].to_numpy(object)[cohort]

    own_rows, own_roles, _ = common.role_labels(outfield, include_draft=False)
    all_rows, all_roles, _ = common.role_labels(outfield, include_draft=True)
    legacy_row = {name: i for i, name in enumerate(legacy["Player"])}
    names = outfield["Player"].to_numpy(object)

    def score(Z_cohort, Z_own, own_r, Z_all, all_r, seed=0):
        return {
            "position_purity@10": metrics.neighbour_purity(Z_cohort, position),
            "family_purity@10": metrics.neighbour_purity(Z_cohort, family),
            "family_probe_F1": metrics.probe_macro_f1(Z_cohort, family, seed),
            "minutes_R2": metrics.leakage_r2(Z_cohort, log_minutes, seed),
            "same_club@10": metrics.same_group_share(Z_cohort, club),
            "role_MAP_own": metrics.retrieval_map(Z_own, own_r),
            "role_MAP_all": metrics.retrieval_map(Z_all, all_r),
        }

    def new_rep(Z):
        return Z[cohort], Z[own_rows], own_roles, Z[all_rows], all_roles

    def legacy_rep(Z):
        own_ok = np.array([names[r] in legacy_row for r in own_rows])
        all_ok = np.array([names[r] in legacy_row for r in all_rows])
        return (Z[legacy_rows],
                Z[[legacy_row[names[r]] for r in own_rows[own_ok]]], own_roles[own_ok],
                Z[[legacy_row[names[r]] for r in all_rows[all_ok]]], all_roles[all_ok])

    rows, per_player = [], {}

    def add(name, dim, reps, stochastic=False):
        scored = [score(*rep, seed=s) for s, rep in enumerate(reps)]
        row = {"representation": name, "dim": dim, "runs": len(reps)}
        for key in scored[0]:
            values = [s[key] for s in scored]
            row[key] = float(np.mean(values))
            if stochastic:
                row[key + "_sd"] = float(np.std(values))
        if stochastic:
            row["seed_stability@10"] = metrics.jaccard_stability([rep[0] for rep in reps])
        rows.append(row)
        nb = metrics.neighbours(reps[0][0], 10)
        per_player[name] = (family[nb] == family[:, None]).mean(axis=1)

    add("original: shipped autoencoder embedding", legacy_Z.shape[1], [legacy_rep(legacy_Z)])
    add("original: 63 input features, standardised", legacy_X.shape[1], [legacy_rep(legacy_X)])
    add("new: performance view only", P.shape[1], [new_rep(P)])
    add("new: attribute view only", A.shape[1], [new_rep(A)])
    add("new: both views, standardised", X.shape[1], [new_rep(X)])
    add(f"new: PCA-{DIM}", DIM, [new_rep(PCA(DIM, random_state=0).fit(X[train]).transform(X))])
    add(f"new: autoencoder-{DIM} (no masking)", DIM,
        [new_rep(autoencoder(dims, DIM, s, False).fit(X[train]).transform(X)) for s in SEEDS], True)
    add(f"new: masked multi-view autoencoder-{DIM}", DIM,
        [new_rep(autoencoder(dims, DIM, s, True).fit(X[train]).transform(X)) for s in SEEDS], True)
    table = pd.DataFrame(rows)
    table.attrs["cohort"] = int(cohort.sum())

    # paired bootstrap: is the purity gain over the original embedding real?
    rng = np.random.default_rng(0)
    base = per_player["original: shipped autoencoder embedding"]
    contrasts = []
    for name in (f"new: PCA-{DIM}", f"new: masked multi-view autoencoder-{DIM}"):
        diff = per_player[name] - base
        boot = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
        contrasts.append({"contrast": f"{name} minus original embedding", "metric": "family_purity@10",
                          "difference": float(diff.mean()), "ci_low": float(np.percentile(boot, 2.5)),
                          "ci_high": float(np.percentile(boot, 97.5)), "n_players": len(diff)})
    return table, pd.DataFrame(contrasts)


def dimension_sweep() -> pd.DataFrame:
    _, views, outfield = common.load_new()
    train = outfield["Min"].to_numpy() >= config.MIN_MINUTES
    cohort = outfield["Min"].to_numpy() >= common.QUALIFIED_MINUTES
    family = outfield["Position"].map(common.POSITION_FAMILY).to_numpy(object)[cohort]
    rows_all, roles_all, _ = common.role_labels(outfield, include_draft=True)
    X = views.combined
    dims = [views.performance.shape[1], views.attributes.shape[1]]
    rows = []
    for d in (2, 4, 8, 12, 16, 24, 32):
        variants = {"PCA": [PCA(d, random_state=0).fit(X[train]).transform(X)],
                    "masked autoencoder": [autoencoder(dims, d, s, True).fit(X[train]).transform(X)
                                           for s in SEEDS[:3]]}
        for model, embeddings in variants.items():
            rows.append({"model": model, "dim": d,
                         "family_purity@10": float(np.mean([metrics.neighbour_purity(Z[cohort], family) for Z in embeddings])),
                         "role_MAP_all": float(np.mean([metrics.retrieval_map(Z[rows_all], roles_all) for Z in embeddings]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- 3. missing view
def _missing_view_scores(gallery, query_full, query_partial, pos_gallery, pos_query, k=10):
    nn = NearestNeighbors(n_neighbors=k).fit(gallery)
    full = nn.kneighbors(query_full, return_distance=False)
    partial = nn.kneighbors(query_partial, return_distance=False)
    jaccard = np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(full, partial)])
    everyone = np.vstack([gallery, query_full])
    dist = np.linalg.norm(query_partial[:, None, :] - everyone[None, :, :], axis=2)
    own = np.arange(len(query_partial)) + len(gallery)
    rank = (dist < dist[np.arange(len(dist)), own][:, None]).sum(axis=1) + 1
    return {"neighbour_overlap@10": float(jaccard),
            "position_purity@10": float((pos_gallery[partial] == pos_query[:, None]).mean()),
            "self_top1": float((rank == 1).mean()), "self_top10": float((rank <= 10).mean()),
            "self_median_rank": float(np.median(rank))}


def missing_view() -> pd.DataFrame:
    """Hold players out, embed them from one view only, compare with their full embedding."""
    _, views, outfield = common.load_new()
    keep = outfield["Min"].to_numpy() >= config.MIN_MINUTES
    X = views.combined[keep]
    position = outfield["Position"].to_numpy(object)[keep]
    n_perf = views.performance.shape[1]
    dims = [n_perf, X.shape[1] - n_perf]
    records = []
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
        pca = PCA(DIM, random_state=0).fit(X[tr])
        models = {"autoencoder (no masking)": autoencoder(dims, DIM, fold, False).fit(X[tr]),
                  "masked multi-view autoencoder": autoencoder(dims, DIM, fold, True).fit(X[tr])}
        for hidden, label, visible_views in ((slice(0, n_perf), "attributes only", [1]),
                                             (slice(n_perf, X.shape[1]), "performance only", [0])):
            observed = np.setdiff1d(np.arange(X.shape[1]), np.arange(X.shape[1])[hidden])
            zero_filled = X[te].copy()
            zero_filled[:, hidden] = 0.0
            ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(X[tr][:, observed], X[tr][:, hidden])
            ridge_filled = X[te].copy()
            ridge_filled[:, hidden] = ridge.predict(X[te][:, observed])
            candidates = {
                "PCA + mean imputation": (pca.transform(X[tr]), pca.transform(X[te]), pca.transform(zero_filled)),
                "PCA + ridge imputation": (pca.transform(X[tr]), pca.transform(X[te]), pca.transform(ridge_filled)),
            }
            for name, model in models.items():
                mask = model.view_mask(len(te), visible_views)
                candidates[name] = (model.transform(X[tr]), model.transform(X[te]), model.transform(X[te], mask))
            for name, (gallery, full, partial) in candidates.items():
                records.append({"available": label, "method": name, "fold": fold,
                                **_missing_view_scores(gallery, full, partial, position[tr], position[te])})
    frame = pd.DataFrame(records)
    mean = frame.groupby(["available", "method"], sort=False).mean(numeric_only=True).drop(columns="fold")
    sd = frame.groupby(["available", "method"], sort=False)["self_top10"].std().rename("self_top10_sd")
    return pd.concat([mean, sd], axis=1).reset_index()


# ---------------------------------------------------------------------------- 4. re-identification
LEGACY_FBREF_COLUMNS = ["MP", "Starts", "Min", "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "Sh", "SoT",
                        "Sh/90", "SoT/90", "G/Sh", "Crs", "Fld", "Fls", "TklW", "Int", "CrdY"]
MIN_SPELL_MINUTES = 360


def reidentification() -> pd.DataFrame:
    """Natural experiment: players who changed club mid-season have two FBref rows.

    A style representation built from one spell should be close to the same player's
    other spell. Each spell is used as a query; the gallery holds the other spell plus
    every single-club outfield player. Two galleries are reported:

    * ``all``            - single-club players with 900+ minutes;
    * ``minutes-matched``- single-club players whose minutes are within 25% of the
                           target spell. This removes the shortcut "both spells are
                           half seasons", which season totals can exploit.
    """
    raw = pd.read_csv(config.FBREF_RAW)
    raw = raw[~raw["Pos"].str.startswith("GK")].copy()
    raw["Born"] = raw["Born"].fillna(-1)
    moved = raw.duplicated(["Player", "Born"], keep=False)
    pairs = []
    for _, group in raw[moved].groupby(["Player", "Born"]):
        top = group.sort_values("Min", ascending=False).iloc[:2]
        if len(top) == 2 and (top["Min"] >= MIN_SPELL_MINUTES).all():
            pairs.append(top)
    first = pd.concat([p.iloc[[0]] for p in pairs])
    second = pd.concat([p.iloc[[1]] for p in pairs])
    single = raw[~moved & (raw["Min"] >= MIN_SPELL_MINUTES)]
    table = pd.concat([single, first, second]).reset_index(drop=True)
    n_single, n_pairs = len(single), len(pairs)
    a_idx = np.arange(n_single, n_single + n_pairs)
    b_idx = a_idx + n_pairs
    minutes = table["Min"].to_numpy(float)
    reference = np.arange(n_single)

    def standardise(values):
        return (values - values[reference].mean(axis=0)) / np.maximum(values[reference].std(axis=0), 1e-9)

    nineties = minutes / 90.0
    plain = pd.DataFrame({k: table[c].to_numpy(float) / nineties for k, c in features.RATE_COUNTS.items()})
    for name, (succ, trials) in features.RATIOS.items():
        plain[name] = (table[succ] / table[trials].replace(0, np.nan)).fillna(0.0).to_numpy()
    shrunk = features.performance_table(table)
    for frame in (plain, shrunk):
        frame[list(features.RATE_COUNTS)] = np.sqrt(frame[list(features.RATE_COUNTS)])
    legacy_cols = table[LEGACY_FBREF_COLUMNS].fillna(table[LEGACY_FBREF_COLUMNS].median())
    with_age = pd.concat([table[["Age"]], legacy_cols], axis=1)
    representations_ = {
        "original FBref features (season totals) + age": standardise(with_age.to_numpy(float)),
        "original FBref features (season totals)": standardise(legacy_cols.to_numpy(float)),
        "per-90 rates, no shrinkage": standardise(plain.to_numpy(float)),
        "per-90 rates + empirical-Bayes shrinkage (this work)": standardise(shrunk.to_numpy(float)),
    }
    queries = list(zip(a_idx, b_idx)) + list(zip(b_idx, a_idx))
    rng = np.random.default_rng(0)
    rows = []
    for name, Z in representations_.items():
        for gallery_name in ("all", "minutes-matched"):
            percentile = []
            for query, target in queries:
                pool = np.arange(n_single)
                if gallery_name == "all":
                    pool = pool[minutes[pool] >= common.QUALIFIED_MINUTES]
                else:
                    pool = pool[np.abs(minutes[pool] - minutes[target]) <= 0.25 * minutes[target]]
                d_pool = np.linalg.norm(Z[pool] - Z[query], axis=1)
                d_true = np.linalg.norm(Z[target] - Z[query])
                percentile.append((d_pool < d_true).mean())
            percentile = np.array(percentile)
            hit = percentile <= 0.05
            boot = hit[rng.integers(0, len(hit), size=(10000, len(hit)))].mean(axis=1)
            rows.append({"representation": name, "gallery": gallery_name, "queries": len(queries),
                         "top5pct_hit_rate": float(hit.mean()),
                         "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                         "median_rank_percentile": float(np.median(percentile))})
    return pd.DataFrame(rows)


def main() -> None:
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    steps = [("data_linkage", data_linkage), ("dimension_sweep", dimension_sweep),
             ("missing_view", missing_view), ("reidentification", reidentification)]
    table, contrasts = representations()
    table.to_csv(common.RESULTS / "representations.csv", index=False)
    contrasts.to_csv(common.RESULTS / "representation_contrasts.csv", index=False)
    print(f"\n== representations (cohort: {table.attrs['cohort']} players) ==")
    print(table.round(3).to_string(index=False))
    print(contrasts.round(3).to_string(index=False))
    for name, fn in steps:
        frame = fn()
        frame.to_csv(common.RESULTS / f"{name}.csv", index=False)
        print(f"\n== {name} ==")
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    main()
