import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from models.prototype_role_engine import (
    get_primary_role
)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "data/final_merged_dataset.csv"
)

embeddings = np.load(
    "models/latent_embeddings.npy"
)

distance_matrix = euclidean_distances(
    embeddings
)

max_dist = np.max(
    distance_matrix
)

similarity_matrix = (
    1 -
    distance_matrix / max_dist
)

# =====================================
# ROLE SYNERGY
# =====================================

ROLE_SYNERGY = {

    "Creative Playmaker": [
        "Ball Winner",
        "Box-to-Box",
        "Deep Playmaker"
    ],

    "Deep Playmaker": [
        "Ball Winner",
        "Box-to-Box",
        "Creative Playmaker"
    ],

    "Ball Winner": [
        "Creative Playmaker",
        "Deep Playmaker",
        "Box-to-Box"
    ],

    "Box-to-Box": [
        "Creative Playmaker",
        "Deep Playmaker",
        "Ball Winner"
    ],

    "Creative Winger": [
        "Target Forward",
        "Poacher",
        "False 9"
    ],

    "Wide Winger": [
        "Target Forward",
        "Poacher"
    ],

    "Inside Forward": [
        "Creative Playmaker",
        "False 9",
        "Target Forward"
    ],

    "False 9": [
        "Creative Winger",
        "Inside Forward"
    ],

    "Poacher": [
        "Creative Winger",
        "Creative Playmaker"
    ],

    "Target Forward": [
        "Creative Winger",
        "Inside Forward"
    ],

    "Ball Playing Defender": [
        "Deep Playmaker",
        "Ball Winner"
    ],

    "Defensive Defender": [
        "Ball Playing Defender"
    ]
}

# =====================================
# POSITION GROUPS
# =====================================

POSITION_GROUPS = {

    "CB": "DEF",
    "RB": "DEF",
    "LB": "DEF",

    "CDM": "MID",
    "CM": "MID",
    "CAM": "MID",

    "RW": "ATT",
    "LW": "ATT",
    "ST": "ATT",

    "RM": "ATT",
    "LM": "ATT",

    "GK": "GK"
}

# =====================================
# POSITION FIT
# =====================================

def position_fit(
    player1,
    player2
):

    pos1 = str(
        player1["Position"]
    )

    pos2 = str(
        player2["Position"]
    )

    group1 = POSITION_GROUPS.get(
        pos1,
        "OTHER"
    )

    group2 = POSITION_GROUPS.get(
        pos2,
        "OTHER"
    )

    if group1 == group2:

        return 0.60

    if {

        group1,
        group2

    } == {

        "MID",
        "ATT"

    }:

        return 1.00

    if {

        group1,
        group2

    } == {

        "MID",
        "DEF"

    }:

        return 0.95

    if {

        group1,
        group2

    } == {

        "DEF",
        "ATT"

    }:

        return 0.70

    return 0.50


# =====================================
# ROLE FIT
# =====================================

def role_compatibility(
    player1,
    player2
):

    role1 = get_primary_role(
        player1
    )

    role2 = get_primary_role(
        player2
    )

    if role1 == role2:

        return 0.30

    if role1 in ROLE_SYNERGY:

        if role2 in ROLE_SYNERGY[
            role1
        ]:

            return 1.00

    if role2 in ROLE_SYNERGY:

        if role1 in ROLE_SYNERGY[
            role2
        ]:

            return 1.00

    return 0.50

# =====================================
# STAT COMPLEMENTARITY
# =====================================

def stat_complementarity(
    player1,
    player2
):

    creativity1 = (

        player1["Vision"]

        +
        player1["Passing"]

        +
        player1["Ball Control"]

    ) / 3

    creativity2 = (

        player2["Vision"]

        +
        player2["Passing"]

        +
        player2["Ball Control"]

    ) / 3

    defending1 = (

        player1["Defending"]

        +
        player1["Interceptions"]

        +
        player1["Standing Tackle"]

    ) / 3

    defending2 = (

        player2["Defending"]

        +
        player2["Interceptions"]

        +
        player2["Standing Tackle"]

    ) / 3

    attack1 = (

        player1["Finishing"]

        +
        player1["Dribbling"]

        +
        player1["Pace"]

    ) / 3

    attack2 = (

        player2["Finishing"]

        +
        player2["Dribbling"]

        +
        player2["Pace"]

    ) / 3

    combo1 = (

        creativity1 * defending2

    )

    combo2 = (

        creativity2 * defending1

    )

    combo3 = (

        attack1 * creativity2

    )

    combo4 = (

        attack2 * creativity1

    )

    score = max(

        combo1,

        combo2,

        combo3,

        combo4

    )

    score /= 10000

    return min(
        score,
        1.0
    )


# =====================================
# MAIN SCORE
# =====================================

def compatibility_score(
    idx1,
    idx2
):

    player1 = df.iloc[idx1]
    player2 = df.iloc[idx2]

    embedding_score = (
        similarity_matrix[idx1][idx2]
    )

    role_score = role_compatibility(
        player1,
        player2
    )

    position_score = position_fit(
        player1,
        player2
    )

    stat_score = stat_complementarity(
        player1,
        player2
    )

    final_score = (

        0.40 * embedding_score

        +

        0.25 * role_score

        +

        0.15 * position_score

        +

        0.20 * stat_score

    )

    return round(
        final_score,
        4
    )


# =====================================
# FIND PLAYER
# =====================================

def find_player(
    player_name
):

    exact = df[
        df["Player"]
        .str.lower()
        ==
        player_name.lower()
    ]

    if len(exact) > 0:

        return exact.index[0]

    partial = df[
        df["Player"]
        .str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(partial) > 0:

        return partial.index[0]

    return None


# =====================================
# BEST PARTNERS
# =====================================

def best_partners(
    player_name,
    top_n=10
):

    player_idx = find_player(
        player_name
    )

    if player_idx is None:

        print(
            "Player not found."
        )

        return

    anchor = df.iloc[
        player_idx
    ]

    anchor_role = get_primary_role(
        anchor
    )

    scores = []

    for idx in range(len(df)):

        if idx == player_idx:
            continue

        score = compatibility_score(

            player_idx,

            idx

        )

        candidate = df.iloc[idx]

        scores.append(

            (

                candidate["Player"],

                candidate["Position"],

                get_primary_role(
                    candidate
                ),

                score

            )

        )

    scores.sort(

        key=lambda x: x[3],

        reverse=True

    )

    print("\n")
    print("=" * 70)

    print(
        f"BEST PARTNERS FOR "
        f"{anchor['Player']}"
    )

    print(
        f"ROLE: {anchor_role}"
    )

    print("=" * 70)

    for player, pos, role, score in scores[:top_n]:

        print(

            f"{player:<30}"

            f"{pos:<8}"

            f"{role:<25}"

            f"{score:.3f}"

        )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    best_partners(
        player_name
    )