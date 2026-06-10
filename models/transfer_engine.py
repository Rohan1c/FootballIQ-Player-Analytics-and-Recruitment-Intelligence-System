import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from compatibility_engine import (
    df
)

from prototype_role_engine import (
    get_primary_role
)

# =====================================
# LOAD EMBEDDINGS
# =====================================

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
# FIND PLAYER
# =====================================

def find_player(player_name):

    exact = df[
        df["Player"].str.lower()
        ==
        player_name.lower()
    ]

    if len(exact) > 0:
        return exact.index[0]

    partial = df[
        df["Player"].str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(partial) > 0:
        return partial.index[0]

    return None


# =====================================
# ROLE MATCH
# =====================================

def role_match(
    role_a,
    role_b
):

    if role_a == role_b:
        return 1.0

    midfield_roles = [

        "Deep Playmaker",
        "Creative Playmaker",
        "Ball Winner",
        "Box-to-Box"

    ]

    attack_roles = [

        "Wide Winger",
        "Creative Winger",
        "Inside Forward",
        "Poacher",
        "Target Forward",
        "False 9"

    ]

    defender_roles = [

        "Ball Playing Defender",
        "Defensive Defender"

    ]

    if (
        role_a in midfield_roles
        and
        role_b in midfield_roles
    ):
        return 0.75

    if (
        role_a in attack_roles
        and
        role_b in attack_roles
    ):
        return 0.75

    if (
        role_a in defender_roles
        and
        role_b in defender_roles
    ):
        return 0.75

    return 0.25


# =====================================
# POSITION MATCH
# =====================================

def position_match(
    pos_a,
    pos_b
):

    if pos_a == pos_b:
        return 1.0

    if pos_a in ["CM", "CAM"] and pos_b in ["CM", "CAM"]:
        return 0.80

    if pos_a in ["RW", "RM"] and pos_b in ["RW", "RM"]:
        return 0.80

    if pos_a in ["LW", "LM"] and pos_b in ["LW", "LM"]:
        return 0.80

    if pos_a in ["CDM", "CM"] and pos_b in ["CDM", "CM"]:
        return 0.75

    return 0.20


# =====================================
# AGE SCORE
# =====================================

def age_score(
    target_age,
    candidate_age
):

    try:

        target_age = float(
            target_age
        )

        candidate_age = float(
            candidate_age
        )

    except:
        return 0.50

    diff = abs(
        target_age -
        candidate_age
    )

    score = max(
        0,
        1 - diff / 15
    )

    return score


# =====================================
# REPLACEMENT SCORE
# =====================================

def replacement_score(
    target_idx,
    candidate_idx
):

    target = df.iloc[
        target_idx
    ]

    candidate = df.iloc[
        candidate_idx
    ]

    embedding_score = (
        similarity_matrix[
            target_idx
        ][
            candidate_idx
        ]
    )

    role_score = role_match(

        get_primary_role(
            target
        ),

        get_primary_role(
            candidate
        )

    )

    position_score = position_match(

        str(
            target["Position"]
        ),

        str(
            candidate["Position"]
        )

    )

    age_fit = age_score(

        target["Age_y"],

        candidate["Age_y"]

    )

    final_score = (

        0.40 * embedding_score

        +

        0.30 * role_score

        +

        0.20 * position_score

        +

        0.10 * age_fit

    )

    return final_score


# =====================================
# FIND REPLACEMENTS
# =====================================

def find_replacements(

    player_name,

    top_n=10,

    younger_only=False

):

    target_idx = find_player(
        player_name
    )

    if target_idx is None:

        print(
            "Player not found."
        )

        return

    target_player = df.iloc[
        target_idx
    ]

    target_age = pd.to_numeric(
        target_player["Age_y"],
        errors="coerce"
    )

    scores = []

    for idx in range(len(df)):

        if idx == target_idx:
            continue

        candidate = df.iloc[idx]

        if younger_only:

            age = pd.to_numeric(

                candidate["Age_y"],

                errors="coerce"

            )

            if pd.isna(age):
                continue

            if age >= target_age:
                continue

        score = replacement_score(

            target_idx,

            idx

        )

        scores.append(

            (

                candidate["Player"],

                candidate["Position"],

                get_primary_role(
                    candidate
                ),

                candidate["Age_y"],

                score

            )

        )

    scores.sort(

        key=lambda x: x[4],

        reverse=True

    )

    print("\n")
    print("=" * 70)
    print("BEST REPLACEMENTS")
    print("=" * 70)

    for rank, player in enumerate(

        scores[:top_n],

        start=1

    ):

        print(

            f"{rank}. "

            f"{player[0]} | "

            f"{player[1]} | "

            f"{player[2]} | "

            f"Age {player[3]} | "

            f"{player[4]:.3f}"

        )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    choice = input(

        "Only younger replacements? (y/n): "

    ).lower()

    younger_only = (
        choice == "y"
    )

    find_replacements(

        player_name,

        top_n=10,

        younger_only=younger_only

    )