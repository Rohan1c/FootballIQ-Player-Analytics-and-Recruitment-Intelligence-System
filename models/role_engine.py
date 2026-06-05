import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/final_merged_dataset.csv"
)

# ==========================
# NORMALIZATION HELPERS
# ==========================

STAT_COLUMNS = [
    "Ast",
    "Gls",
    "G+A",
    "Compl",
    "PPM",
    "Int",
    "TklW",
    "Sh/90",
    "SoT/90",
    "Crs"
]

for col in STAT_COLUMNS:

    if col in df.columns:

        mn = df[col].min()
        mx = df[col].max()

        if mx != mn:

            df[f"{col}_norm"] = (
                (df[col] - mn)
                /
                (mx - mn)
            ) * 100

        else:

            df[f"{col}_norm"] = 0


# ==========================
# ROLE SCORING
# ==========================

def get_role_scores(player):

    scores = {}

    # ----------------------
    # DEEP PLAYMAKER
    # ----------------------

    scores["Deep Playmaker"] = (

        0.15 * player["Passing"] +
        0.15 * player["Vision"] +
        0.15 * player["Long Passing"] +
        0.10 * player["Composure"] +

        0.10 * player["Defending"] +
        0.10 * player["Interceptions"] +

        0.10 * player["Ast_norm"] +
        0.10 * player["Compl_norm"] +
        0.05 * player["PPM_norm"]

    )

    # ----------------------
    # CREATIVE PLAYMAKER
    # ----------------------

    scores["Creative Playmaker"] = (

        0.20 * player["Vision"] +
        0.20 * player["Passing"] +
        0.20 * player["Dribbling"] +
        0.15 * player["Ball Control"] +

        0.10 * player["Ast_norm"] +
        0.10 * player["G+A_norm"] +

        0.05 * player["Agility"]

    )

    # ----------------------
    # BOX TO BOX
    # ----------------------

    scores["Box-to-Box"] = (

        0.15 * player["Passing"] +
        0.15 * player["Defending"] +
        0.15 * player["Physicality"] +
        0.15 * player["Stamina"] +

        0.10 * player["Aggression"] +
        0.10 * player["TklW_norm"] +
        0.10 * player["Int_norm"] +

        0.10 * player["PPM_norm"]

    )

    # ----------------------
    # BALL WINNER
    # ----------------------

    scores["Ball Winner"] = (

        0.20 * player["Defending"] +
        0.20 * player["Interceptions"] +
        0.20 * player["Standing Tackle"] +

        0.10 * player["Strength"] +
        0.10 * player["Aggression"] +

        0.10 * player["Int_norm"] +
        0.10 * player["TklW_norm"]

    )

    # ----------------------
    # BALL PLAYING DEFENDER
    # ----------------------

    scores["Ball Playing Defender"] = (

        0.20 * player["Defending"] +
        0.15 * player["Passing"] +
        0.15 * player["Vision"] +

        0.15 * player["Composure"] +
        0.10 * player["Long Passing"] +

        0.10 * player["Int_norm"] +
        0.10 * player["Compl_norm"] +
        0.05 * player["TklW_norm"]

    )

    # ----------------------
    # DEFENSIVE DEFENDER
    # ----------------------

    scores["Defensive Defender"] = (

        0.25 * player["Defending"] +
        0.20 * player["Interceptions"] +
        0.20 * player["Standing Tackle"] +

        0.10 * player["Sliding Tackle"] +
        0.10 * player["Strength"] +

        0.10 * player["TklW_norm"] +
        0.05 * player["Int_norm"]

    )

    # ----------------------
    # CREATIVE WINGER
    # ----------------------

    scores["Creative Winger"] = (

        0.15 * player["Pace"] +
        0.20 * player["Dribbling"] +
        0.15 * player["Vision"] +

        0.15 * player["Crossing"] +
        0.10 * player["Agility"] +

        0.10 * player["Ast_norm"] +
        0.10 * player["Crs_norm"] +
        0.05 * player["G+A_norm"]

    )

    # ----------------------
    # INSIDE FORWARD
    # ----------------------

    scores["Inside Forward"] = (

        0.20 * player["Pace"] +
        0.20 * player["Dribbling"] +
        0.20 * player["Finishing"] +

        0.15 * player["Positioning"] +
        0.10 * player["Shot Power"] +

        0.10 * player["G+A_norm"] +
        0.05 * player["SoT/90_norm"]

    )

    # ----------------------
    # TARGET FORWARD
    # ----------------------

    scores["Target Forward"] = (

        0.20 * player["Finishing"] +
        0.15 * player["Strength"] +
        0.15 * player["Heading Accuracy"] +

        0.15 * player["Positioning"] +
        0.10 * player["Physicality"] +

        0.15 * player["Gls_norm"] +
        0.10 * player["Sh/90_norm"]

    )

    # ----------------------
    # POACHER
    # ----------------------

    scores["Poacher"] = (

        0.25 * player["Finishing"] +
        0.20 * player["Positioning"] +

        0.10 * player["Shot Power"] +
        0.10 * player["Composure"] +

        0.15 * player["Gls_norm"] +
        0.10 * player["SoT/90_norm"] +
        0.10 * player["Sh/90_norm"]

    )

    return scores


def get_primary_role(player):

    scores = get_role_scores(player)

    return max(
        scores,
        key=scores.get
    )


def get_top_roles(player, top_n=5):

    scores = get_role_scores(player)

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]


if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    row = df[
        df["Player"].str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(row) == 0:

        print("Player not found.")
        exit()

    player = row.iloc[0]

    print(
        f"\nPlayer: {player['Player']}"
    )

    print(
        f"Position: {player['Pos']}"
    )

    print("\nTop Roles:\n")

    for role, score in get_top_roles(player):

        print(
            f"{role:<25} {round(score,2)}"
        )

    print(
        "\nPrimary Role:"
    )

    print(
        get_primary_role(player)
    )