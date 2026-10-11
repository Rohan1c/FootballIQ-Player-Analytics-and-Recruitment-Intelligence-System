"""Benchmark of tactical-role classification.

Three questions, three tables under ``evaluation/results/``:

  roles_original_engine.csv  How much of the originally reported accuracy survives once
                             leaked and trivially correct players are removed, and on
                             players the engine was never tuned on?
  roles_heldout.csv          On those held-out players, how does a classifier with no
                             hand-tuned weights compare with the original engine?
  roles_cross_validation.csv With every available label, what accuracy does
                             cross-validation give, and how does it grow with labels?
  roles_career_history.csv   Does adding a player's earlier seasons (which have passing and
                             pressing data) help, for the players who have such seasons?

Label sets (see evaluation/README.md):
  own    evaluation/role_labels.csv                   - the project's own 58 outfield labels
  draft  evaluation/role_labels_draft_unverified.csv  - 153 extra labels drafted by an AI
         assistant before any prediction for those players was looked at. They are NOT
         verified by a human; results that use them are marked and must be re-run after review.

Run from the repository root:  python -m evaluation.benchmark_roles
"""
from __future__ import annotations

import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from evaluation import common, metrics
from footballiq import config, corpus
from footballiq.roles import (ORIGINAL_POSITION_ROLE_MAP, POSITION_ROLE_MAP, REGULARISATION,
                              CareerAwareRoleClassifier, RoleClassifier, apply_gate)

warnings.filterwarnings("ignore")
C_GRID = [0.03, 0.1, 0.3, 1.0]

# The original prototypes were looked up with a substring search; these are the players
# the author intended, used for the "exemplars fixed" variant of the original engine.
INTENDED_EXEMPLARS = {
    "Deep Playmaker": ["Rodri", "Hakan Çalhanoğlu", "Enzo Fernández"],
    "Creative Playmaker": ["Kevin De Bruyne", "Pedri", "Martin Ødegaard"],
    "Ball Winner": ["Moisés Caicedo", "Aurélien Tchouaméni", "Declan Rice", "Manuel Ugarte"],
    "Box-to-Box": ["Federico Valverde", "Ryan Gravenberch"],
    "Wide Winger": ["Raphinha", "Bukayo Saka", "Adama Traoré"],
    "Creative Winger": ["Florian Wirtz", "Cole Palmer", "Jamal Musiala"],
    "Inside Forward": ["Khvicha Kvaratskhelia", "Mohamed Salah"],
    "Poacher": ["Erling Haaland", "Robert Lewandowski", "Viktor Gyökeres"],
    "Target Forward": ["Romelu Lukaku", "Dušan Vlahović"],
    "False 9": ["Antoine Griezmann"],
    "Ball Playing Defender": ["Virgil van Dijk", "William Saliba", "Rúben Dias"],
    "Defensive Defender": ["Nikola Milenković", "Fikayo Tomori", "Jules Koundé"],
}


def _summary(name: str, correct, **extra) -> dict:
    correct = np.asarray(correct, float)
    low, high = metrics.bootstrap_ci(correct)
    return {"evaluation": name, "n": len(correct), "accuracy": float(correct.mean()),
            "ci_low": low, "ci_high": high, **extra}


# ---------------------------------------------------------------------------- original engine
def _legacy_engine():
    with contextlib.redirect_stdout(io.StringIO()):
        import legacy.models.prototype_role_engine as engine
    return engine


def _legacy_predict(engine, names) -> np.ndarray:
    out = []
    for name in names:
        row = engine.find_player(name)
        out.append(engine.get_primary_role(row) if row is not None else None)
    return np.array(out, dtype=object)


def original_engine(draft: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    engine = _legacy_engine()
    validation = pd.read_csv(config.LEGACY_DIR / "role_validation_labels.csv", header=None,
                             names=["player", "role"], skipinitialspace=True).dropna()
    rows = [(name, role, engine.find_player(name)) for name, role in zip(validation["player"], validation["role"])]
    rows = [(name, role, row) for name, role, row in rows if row is not None]
    truth = np.array([role for _, role, _ in rows], dtype=object)
    predicted = np.array([engine.get_primary_role(row) for _, _, row in rows], dtype=object)
    matched = np.array([row["Player"] for _, _, row in rows], dtype=object)
    is_gk = np.array([row["Position"] == "GK" for _, _, row in rows])

    # players that actually went into the prototypes (substring search, first hit)
    exemplars = set()
    for names in engine.ROLE_EXEMPLARS.values():
        for name in names:
            hits = engine.df[engine.df["Player"].str.contains(name, case=False, na=False)]
            if len(hits):
                exemplars.add(hits.iloc[0]["Player"])
    leaked = np.isin(matched, list(exemplars))
    allowed = [engine.POSITION_ROLE_MAP.get(engine.normalize_position(row["Position"]), []) for _, _, row in rows]
    chance = np.array([(t in a) / max(len(a), 1) for t, a in zip(truth, allowed)])

    draft_pred = _legacy_predict(engine, draft["player"])
    draft_truth = draft["role"].to_numpy(object)
    clean = ~leaked & ~is_gk
    table = pd.DataFrame([
        _summary("as reported: all validation players", predicted == truth, note="in-sample; includes exemplars and goalkeepers"),
        _summary("validation players that are prototype exemplars", (predicted == truth)[leaked], note="leak: test players define the prototypes"),
        _summary("validation goalkeepers", (predicted == truth)[is_gk], note="correct by rule"),
        _summary("validation, excluding exemplars and goalkeepers", (predicted == truth)[clean], note="still in-sample: weights were hand-tuned on this list"),
        _summary("random choice among the roles the position gate allows", chance[~is_gk], note="chance level"),
        _summary("held-out draft labels (never used for tuning)", draft_pred == draft_truth, note="out-of-sample; labels unverified"),
    ])
    return table, draft_pred


def original_engine_fixed(draft: pd.DataFrame) -> np.ndarray:
    """Original scoring rules, but prototypes built from the intended exemplars."""
    engine = _legacy_engine()
    saved = dict(engine.ROLE_PROTOTYPES)
    try:
        engine.ROLE_PROTOTYPES.clear()
        for role, names in INTENDED_EXEMPLARS.items():
            members = engine.df[engine.df["Player"].isin(names)]
            engine.ROLE_PROTOTYPES[role] = members[engine.FEATURE_COLS].to_numpy(float).mean(axis=0)
        return _legacy_predict(engine, draft["player"])
    finally:
        engine.ROLE_PROTOTYPES.clear()
        engine.ROLE_PROTOTYPES.update(saved)


# ---------------------------------------------------------------------------- learned classifiers
class NearestPrototype:
    """Class centroid in feature space; the data-driven version of the original idea."""

    def fit(self, X, y):
        self.classes_ = list(np.unique(y))
        self.centroids = np.array([X[np.asarray(y, object) == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X, positions):
        dist = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        proba = apply_gate(np.exp(-dist), positions, self.classes_)
        return np.asarray(self.classes_, object)[proba.argmax(axis=1)]


def _select_c(X, y, positions, seed: int) -> float:
    """Inner cross-validation to choose the regularisation strength (no test data involved)."""
    counts = pd.Series(y).value_counts()
    strata = np.where(pd.Series(y).map(counts).to_numpy() < 3, "rare", y)
    best_c, best_acc = REGULARISATION, -1.0
    for c in C_GRID:
        hits = []
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, strata):
            model = RoleClassifier(C=c).fit(X[tr], y[tr])
            hits.append(np.mean(model.predict(X[va], positions[va]) == y[va]))
        if np.mean(hits) > best_acc:
            best_c, best_acc = c, float(np.mean(hits))
    return best_c


CAREER = "experiment: + earlier rich-data seasons where available"


def cross_validated(X, y, positions, H, has_history, repeats: int = 10):
    counts = pd.Series(y).value_counts()
    strata = np.where(pd.Series(y).map(counts).to_numpy() < 5, "rare", y)
    fixed_name = f"logistic regression (C={REGULARISATION}) + gate"
    scores = {"logistic regression (C chosen by inner CV) + gate": [],
              fixed_name: [],
              f"logistic regression (C={REGULARISATION}) + original gate": [],
              "logistic regression, no gate": [], "nearest prototype + gate": [], CAREER: []}
    subset = {(name, group): [] for name in (fixed_name, CAREER) for group in ("with history", "without history")}
    for rep in range(repeats):
        fold_hits = {k: np.zeros(len(y), dtype=bool) for k in scores}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, strata):
            c = _select_c(X[tr], y[tr], positions[tr], seed=rep)
            nested = RoleClassifier(C=c).fit(X[tr], y[tr]).predict(X[te], positions[te])
            fixed = RoleClassifier().fit(X[tr], y[tr])
            proto = NearestPrototype().fit(X[tr], y[tr]).predict(X[te], positions[te])
            original_gate = RoleClassifier(gate=ORIGINAL_POSITION_ROLE_MAP).fit(X[tr], y[tr])
            career = CareerAwareRoleClassifier().fit(X[tr], H[tr], has_history[tr], y[tr])
            fold_hits["logistic regression (C chosen by inner CV) + gate"][te] = nested == y[te]
            fold_hits[fixed_name][te] = fixed.predict(X[te], positions[te]) == y[te]
            fold_hits[f"logistic regression (C={REGULARISATION}) + original gate"][te] = \
                original_gate.predict(X[te], positions[te]) == y[te]
            fold_hits["logistic regression, no gate"][te] = fixed.predict(X[te]) == y[te]
            fold_hits["nearest prototype + gate"][te] = proto == y[te]
            fold_hits[CAREER][te] = career.predict(X[te], H[te], has_history[te], positions[te]) == y[te]
        for k in scores:
            scores[k].append(fold_hits[k].mean())
        for name in (fixed_name, CAREER):
            subset[(name, "with history")].append(fold_hits[name][has_history].mean())
            subset[(name, "without history")].append(fold_hits[name][~has_history].mean())
    table = pd.DataFrame([{"classifier": k, "labels": len(y), "protocol": f"{repeats}x repeated 5-fold CV",
                           "accuracy": float(np.mean(v)), "sd_over_repeats": float(np.std(v))}
                          for k, v in scores.items()])
    by_history = pd.DataFrame([{"classifier": name, "labelled_players": group,
                                "n": int(has_history.sum() if group == "with history" else (~has_history).sum()),
                                "accuracy": float(np.mean(v)), "sd_over_repeats": float(np.std(v))}
                               for (name, group), v in subset.items()])
    # leave-one-out predictions of the shipped configuration: one prediction per player
    loo = np.empty(len(y), dtype=object)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        loo[i] = RoleClassifier().fit(X[keep], y[keep]).predict(X[[i]], positions[[i]])[0]
    return table, by_history, loo


def learning_curve(X, y, positions, sizes=(30, 58, 90, 120, 150, 180), repeats: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for n in sizes:
        hits = {"logistic regression + gate": [], "nearest prototype + gate": []}
        for _ in range(repeats):
            perm = rng.permutation(len(y))
            tr, te = perm[:n], perm[n:]
            if len(np.unique(y[tr])) < 2:
                continue
            hits["logistic regression + gate"].append(
                np.mean(RoleClassifier().fit(X[tr], y[tr]).predict(X[te], positions[te]) == y[te]))
            hits["nearest prototype + gate"].append(
                np.mean(NearestPrototype().fit(X[tr], y[tr]).predict(X[te], positions[te]) == y[te]))
        for name, values in hits.items():
            rows.append({"classifier": name, "training_labels": n,
                         "accuracy": float(np.mean(values)), "sd": float(np.std(values))})
    return pd.DataFrame(rows)


def main() -> None:
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    players, views, outfield = common.load_new()
    X = views.combined
    positions = outfield["Position"].to_numpy(object)
    has_history, H = corpus.career_features(players)
    has_history, H = has_history[views.index], H[views.index]
    own_rows, own_roles, _ = common.role_labels(outfield, include_draft=False)
    all_rows, all_roles, source = common.role_labels(outfield, include_draft=True)
    draft = pd.read_csv(config.ROLE_LABELS_DRAFT)
    is_draft = source == "draft_unverified"
    draft_rows, draft_roles = all_rows[is_draft], all_roles[is_draft]
    assert list(outfield["Player"].to_numpy()[draft_rows]) == list(draft["player"])

    # ---- 1. original engine: reported vs honest
    original, legacy_pred = original_engine(draft)
    original.to_csv(common.RESULTS / "roles_original_engine.csv", index=False)

    # ---- 2. held-out comparison: everything below is trained on the own labels only
    gate_sizes = np.array([len(ORIGINAL_POSITION_ROLE_MAP[p]) for p in positions[draft_rows]])
    reachable = np.array([r in ORIGINAL_POSITION_ROLE_MAP[p] for r, p in zip(draft_roles, positions[draft_rows])])
    fixed_pred = original_engine_fixed(draft)
    logistic = RoleClassifier().fit(X[own_rows], own_roles).predict(X[draft_rows], positions[draft_rows])
    prototype = NearestPrototype().fit(X[own_rows], own_roles).predict(X[draft_rows], positions[draft_rows])
    heldout = pd.DataFrame([
        _summary("random choice within the original position gate", reachable / gate_sizes, trained_on="-"),
        _summary("original rule engine (as shipped)", legacy_pred == draft_roles, trained_on="hand-tuned on the validation list"),
        _summary("original rule engine, exemplars fixed", fixed_pred == draft_roles, trained_on="hand-tuned on the validation list"),
        _summary("nearest prototype + gate (no hand-tuned weights)", prototype == draft_roles, trained_on="58 own labels"),
        _summary("logistic regression + gate (no hand-tuned weights)", logistic == draft_roles, trained_on="58 own labels"),
    ])

    # ---- 3. all labels, cross-validation
    cv, by_history, loo = cross_validated(X[all_rows], all_roles, positions[all_rows],
                                         H[all_rows], has_history[all_rows])
    by_history.to_csv(common.RESULTS / "roles_career_history.csv", index=False)
    loo_correct = loo == all_roles
    cv.to_csv(common.RESULTS / "roles_cross_validation.csv", index=False)
    heldout = pd.concat([heldout, pd.DataFrame([
        _summary("logistic regression + gate, leave-one-out over all labels (draft players only)",
                 loo_correct[is_draft], trained_on="210 labels (own + draft)")])], ignore_index=True)
    heldout.to_csv(common.RESULTS / "roles_heldout.csv", index=False)

    reach = pd.DataFrame([
        {"gate": name, "labelled_players": len(all_roles),
         "true_role_allowed": float(np.mean([r in gate[p] for r, p in zip(all_roles, positions[all_rows])]))}
        for name, gate in (("original", ORIGINAL_POSITION_ROLE_MAP), ("revised (RM/LM may be Inside Forward)", POSITION_ROLE_MAP))])
    reach.to_csv(common.RESULTS / "roles_gate_reachability.csv", index=False)

    a, b = legacy_pred == draft_roles, loo_correct[is_draft]
    only_new, only_old = int((~a & b).sum()), int((a & ~b).sum())
    mcnemar = pd.DataFrame([{
        "comparison": "logistic regression (leave-one-out) vs original engine, 153 held-out players",
        "only_new_correct": only_new, "only_original_correct": only_old,
        "exact_mcnemar_p": float(binomtest(only_new, only_new + only_old, 0.5).pvalue)}])
    mcnemar.to_csv(common.RESULTS / "roles_mcnemar.csv", index=False)

    curve = learning_curve(X[all_rows], all_roles, positions[all_rows])
    curve.to_csv(common.RESULTS / "roles_learning_curve.csv", index=False)

    per_role = pd.DataFrame({
        "n": pd.Series(all_roles).value_counts(),
        "accuracy_leave_one_out": pd.Series(loo_correct).groupby(all_roles).mean(),
    }).reset_index(names="role")
    per_role.to_csv(common.RESULTS / "roles_per_role.csv", index=False)

    # ---- figures
    labels = sorted(set(all_roles))
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix(all_roles, loo, labels=labels), display_labels=labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"Role classifier, leave-one-out over {len(all_roles)} labelled players")
    plt.tight_layout()
    plt.savefig(common.RESULTS / "role_confusion_matrix.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, group in curve.groupby("classifier"):
        ax.errorbar(group["training_labels"], group["accuracy"], yerr=group["sd"], marker="o", capsize=3, label=name)
    ax.axhline(float((legacy_pred == draft_roles).mean()), color="grey", linestyle="--",
               label="original rule engine (held-out)")
    ax.set_xlabel("number of labelled players used for training")
    ax.set_ylabel("accuracy on the remaining labelled players")
    ax.set_title("Role accuracy grows with labels, then flattens")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(common.RESULTS / "role_learning_curve.png", dpi=200)
    plt.close()

    for title, frame in [("original engine: reported vs honest", original), ("held-out players", heldout),
                         ("cross-validation, all labels", cv), ("by career history", by_history), ("gate", reach), ("McNemar", mcnemar),
                         ("learning curve", curve), ("per role", per_role)]:
        print(f"\n== {title} ==")
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 90)
    main()
