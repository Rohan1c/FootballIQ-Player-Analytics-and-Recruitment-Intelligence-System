"""Tools for building a role-label set that can be defended in a paper.

    python -m evaluation.labelling make     write evaluation/labelling_sheet.csv
    python -m evaluation.labelling score    agreement between the two annotators

``make`` draws a sample of regular starters, stratified by position, with two empty
columns. Two people fill in one column each WITHOUT looking at the model's output or
at each other's answers. ``score`` then reports raw agreement and Cohen's kappa, and
writes the players both annotators agree on to evaluation/role_labels_agreed.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from footballiq import config
from footballiq.roles import ROLES

SHEET = config.ROOT / "evaluation" / "labelling_sheet.csv"
AGREED = config.ROOT / "evaluation" / "role_labels_agreed.csv"
PER_POSITION = 30
MIN_MINUTES = 1350            # about fifteen full matches


def make() -> pd.DataFrame:
    players = pd.read_csv(config.PLAYERS_CSV)
    pool = players[(players["Position"] != "GK") & (players["Min"] >= MIN_MINUTES)]
    draft = pd.read_csv(config.ROLE_LABELS_DRAFT)
    rng = np.random.default_rng(config.SEED)
    chosen = []
    for position, group in pool.groupby("Position"):
        must = group[group["Player"].isin(draft["player"])]          # so the draft labels get checked
        rest = group.drop(must.index)
        extra = rest.sample(min(len(rest), max(PER_POSITION - len(must), 10)), random_state=int(rng.integers(1e9)))
        chosen.append(pd.concat([must, extra]))
    sheet = pd.concat(chosen)[["Player", "Born", "Squad", "Position", "Age", "Min"]]
    sheet = sheet.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)   # shuffled order
    sheet["annotator_1"] = ""
    sheet["annotator_2"] = ""
    sheet.to_csv(SHEET, index=False)
    print(f"wrote {SHEET.relative_to(config.ROOT)}: {len(sheet)} players")
    print("allowed roles: " + ", ".join(ROLES))
    return sheet


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, object), np.asarray(b, object)
    observed = float(np.mean(a == b))
    labels = np.union1d(a, b)
    expected = float(sum(np.mean(a == l) * np.mean(b == l) for l in labels))
    return (observed - expected) / (1.0 - expected) if expected < 1 else 1.0


def score() -> None:
    sheet = pd.read_csv(SHEET).fillna("")
    done = sheet[(sheet["annotator_1"] != "") & (sheet["annotator_2"] != "")]
    if done.empty:
        print("No player has both annotator columns filled in yet.")
        return
    unknown = set(done["annotator_1"]) | set(done["annotator_2"])
    unknown -= set(ROLES)
    if unknown:
        print(f"Unknown role names (fix the spelling): {sorted(unknown)}")
        return
    agree = done["annotator_1"] == done["annotator_2"]
    print(f"players labelled by both : {len(done)}")
    print(f"raw agreement            : {agree.mean():.3f}")
    print(f"Cohen's kappa            : {cohen_kappa(done['annotator_1'], done['annotator_2']):.3f}")
    out = done[agree].rename(columns={"Player": "player", "Born": "born", "annotator_1": "role",
                                      "Position": "position", "Min": "minutes"})
    out = out.assign(source="two_annotators_agree")[["player", "born", "role", "source", "position", "minutes"]]
    out.to_csv(AGREED, index=False)
    print(f"wrote {AGREED.relative_to(config.ROOT)}: {len(out)} agreed labels")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    {"make": make, "score": score}.get(command, lambda: print(__doc__))()
