import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "data/final_merged_dataset.csv"
)


# =====================================
# FEATURES
# =====================================

FEATURE_COLS = [

    "Pace",
    "Shooting",
    "Passing",
    "Dribbling",
    "Defending",
    "Physicality",

    "Vision",
    "Crossing",
    "Finishing",
    "Ball Control",

    "Interceptions",
    "Standing Tackle",

    "Strength",
    "Aggression"

]


# =====================================
# NORMALIZE
# =====================================

scaler = MinMaxScaler()

df[FEATURE_COLS] = scaler.fit_transform(
    df[FEATURE_COLS]
)


# =====================================
# POSITION GATING
# =====================================

POSITION_ROLE_MAP = {

    "GK": [
        "Goalkeeper"
    ],

    "CB": [
        "Ball Playing Defender",
        "Defensive Defender"
    ],

    "RB": [
        "Ball Playing Defender",
        "Wide Winger"
    ],

    "LB": [
        "Ball Playing Defender",
        "Wide Winger"
    ],

    "CDM": [
        "Deep Playmaker",
        "Ball Winner",
        "Box-to-Box"
    ],

    "CM": [
        "Deep Playmaker",
        "Ball Winner",
        "Box-to-Box",
        "Creative Playmaker"
    ],

    "CAM": [
        "Creative Playmaker",
        "False 9",
        "Box-to-Box"
    ],

    "RW": [
        "Wide Winger",
        "Creative Winger",
        "Inside Forward"
    ],

    "LW": [
        "Wide Winger",
        "Creative Winger",
        "Inside Forward"
    ],

    "RM": [
        "Wide Winger",
        "Creative Winger",
        "Box-to-Box"
    ],

    "LM": [
        "Wide Winger",
        "Creative Winger",
        "Box-to-Box"
    ],

    "ST": [
        "Poacher",
        "Target Forward",
        "False 9"
    ]
}


# =====================================
# ROLE WEIGHTS
# =====================================

ROLE_WEIGHTS = {

    "Deep Playmaker": [
        "Passing",
        "Vision",
        "Ball Control"
    ],

    "Creative Playmaker": [
        "Passing",
        "Vision",
        "Dribbling",
        "Ball Control"
    ],

    "Ball Winner": [
        "Defending",
        "Interceptions",
        "Standing Tackle",
        "Aggression"
    ],

    "Box-to-Box": [
        "Passing",
        "Defending",
        "Physicality",
        "Stamina"
    ],

    "Wide Winger": [
        "Pace",
        "Crossing",
        "Dribbling"
    ],

    "Creative Winger": [
        "Dribbling",
        "Vision",
        "Passing"
    ],

    "Inside Forward": [
        "Pace",
        "Finishing",
        "Dribbling"
    ],

    "Poacher": [
        "Finishing",
        "Positioning",
        "Shooting"
    ],

    "Target Forward": [
        "Strength",
        "Physicality",
        "Finishing"
    ],

    "False 9": [
        "Passing",
        "Vision",
        "Ball Control"
    ],

    "Ball Playing Defender": [
        "Passing",
        "Defending",
        "Interceptions"
    ],

    "Defensive Defender": [
        "Defending",
        "Standing Tackle",
        "Strength"
    ]
}


# =====================================
# EXEMPLARS
# =====================================

ROLE_EXEMPLARS = {

    "Deep Playmaker": [
        "Rodri",
        "Kimmich"
    ],

    "Creative Playmaker": [
        "Kevin De Bruyne",
        "Pedri",
        "Bernardo Silva"
    ],

    "Ball Winner": [
        "Caicedo",
        "Declan Rice",
        "Tchouameni"
    ],

    "Box-to-Box": [
        "Valverde",
        "Bellingham"
    ],

    "Wide Winger": [
        "Raphinha",
        "Saka"
    ],

    "Creative Winger": [
        "Wirtz",
        "Palmer"
    ],

    "Inside Forward": [
        "Son",
        "Kvaratskhelia"
    ],

    "Poacher": [
        "Haaland",
        "Kane"
    ],

    "Target Forward": [
        "Lukaku",
        "Osimhen"
    ],

    "False 9": [
        "Benzema",
        "Griezmann"
    ],

    "Ball Playing Defender": [
        "Van Dijk",
        "Saliba"
    ],

    "Defensive Defender": [
        "Milenkovic",
        "Tomori"
    ]
}

# =====================================
# BUILD PROTOTYPES
# =====================================

def build_prototypes(verbose=False):

    prototypes = {}

    if verbose:
        print("\nROLE CENTROIDS\n")
    

    for role, names in ROLE_EXEMPLARS.items():

        vectors = []

        found = 0

        for name in names:

            player_rows = df[
                df["Player"]
                .str.contains(
                    name,
                    case=False,
                    na=False
                )
            ]

            if len(player_rows) == 0:
                continue

            found += 1

            player = player_rows.iloc[0]

            vectors.append(
                player[
                    FEATURE_COLS
                ].values
            )

        if verbose:
            print(
                f"{role}: "
                f"{found}/{len(names)}"
            )

        if len(vectors) == 0:
            continue

        centroid = np.mean(
            vectors,
            axis=0
        )

        prototypes[
            role
        ] = centroid

    return prototypes


# =====================================
# GLOBAL PROTOTYPES
# =====================================

ROLE_PROTOTYPES = (
    build_prototypes(verbose=False)
)


# =====================================
# ROLE SCORES
# =====================================

def get_role_scores(player):

    position = str(
        player["Position"]
    )

    if position == "GK":

        return {
            "Goalkeeper": 1.0
        }

    allowed_roles = (
        POSITION_ROLE_MAP.get(
            position,
            list(
                ROLE_PROTOTYPES.keys()
            )
        )
    )

    player_vector = player[
        FEATURE_COLS
    ].values.astype(
        float
    )

    scores = {}

    for role in allowed_roles:

        if role not in ROLE_PROTOTYPES:
            continue

        prototype = ROLE_PROTOTYPES[
            role
        ]

        similarity = cosine_similarity(
            player_vector.reshape(1, -1),
            prototype.reshape(1, -1)
        )[0][0]

        # -------------------------
        # ROLE SPECIFIC BOOSTS
        # -------------------------

        bonus = 0

        if role == "Deep Playmaker":

            bonus += (
                player["Passing"] * 0.10
            )

            bonus += (
                player["Vision"] * 0.10
            )

        elif role == "Creative Playmaker":

            bonus += (
                player["Vision"] * 0.10
            )

            bonus += (
                player["Dribbling"] * 0.10
            )

        elif role == "Ball Winner":

            bonus += (
                player["Defending"] * 0.10
            )

            bonus += (
                player["Interceptions"] * 0.10
            )

        elif role == "Box-to-Box":

            bonus += (
                player["Physicality"] * 0.08
            )

            bonus += (
                player["Passing"] * 0.08
            )

        elif role == "Wide Winger":

            bonus += (
                player["Pace"] * 0.10
            )

            bonus += (
                player["Crossing"] * 0.10
            )

        elif role == "Creative Winger":

            bonus += (
                player["Dribbling"] * 0.10
            )

            bonus += (
                player["Vision"] * 0.10
            )

        elif role == "Inside Forward":

            bonus += (
                player["Finishing"] * 0.10
            )

            bonus += (
                player["Pace"] * 0.10
            )

        elif role == "Poacher":

            bonus += (
                player["Finishing"] * 0.15
            )

        elif role == "Target Forward":

            bonus += (
                player["Strength"] * 0.15
            )

        elif role == "False 9":

            bonus += (
                player["Passing"] * 0.10
            )

            bonus += (
                player["Vision"] * 0.10
            )

        elif role == "Ball Playing Defender":

            bonus += (
                player["Passing"] * 0.10
            )

            bonus += (
                player["Defending"] * 0.08
            )

        elif role == "Defensive Defender":

            bonus += (
                player["Defending"] * 0.12
            )

            bonus += (
                player["Strength"] * 0.08
            )

        scores[role] = float(
            similarity + bonus
        )

    return scores


# =====================================
# PRIMARY ROLE
# =====================================

def get_primary_role(player):

    scores = get_role_scores(
        player
    )

    if len(scores) == 0:

        return "Undefined"

    sorted_scores = sorted(

        scores.items(),

        key=lambda x: x[1],

        reverse=True

    )

    best_role = sorted_scores[0][0]
    best_score = sorted_scores[0][1]

    # only one role available

    if len(sorted_scores) == 1:

        return best_role

    second_score = (
        sorted_scores[1][1]
    )

    confidence_gap = (
        best_score -
        second_score
    )

    # if roles are extremely close,
    # still return best role,
    # but avoid hard 0.80 threshold

    if confidence_gap < 0.01:

        return best_role

    return best_role


# =====================================
# FIND PLAYER
# =====================================

def find_player(player_name):

    exact = df[
        df["Player"]
        .str.lower()
        ==
        player_name.lower()
    ]

    if len(exact) > 0:

        return exact.iloc[0]

    partial = df[
        df["Player"]
        .str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(partial) > 0:

        return partial.iloc[0]

    return None


# =====================================
# TESTER
# =====================================

if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    player = find_player(
        player_name
    )

    if player is None:

        print(
            "Player not found."
        )

        exit()

    scores = (
        get_role_scores(
            player
        )
    )

    sorted_scores = sorted(

        scores.items(),

        key=lambda x: x[1],

        reverse=True

    )

    print("\n")
    print(
        f"Player: "
        f"{player['Player']}"
    )

    print(
        f"Position: "
        f"{player['Position']}"
    )

    print(
        "\nTop Prototype Roles:\n"
    )

    for role, score in (
        sorted_scores[:5]
    ):

        print(
            f"{role:<25}"
            f"{score:.4f}"
        )

    print("\n")
    print(
        "Primary Role:"
    )

    print(
        get_primary_role(
            player
        )
    )