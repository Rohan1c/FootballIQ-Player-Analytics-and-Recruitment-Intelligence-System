import pandas as pd

from compatibility_engine import (
    df,
    compatibility_score
)

from prototype_role_engine import (
    get_primary_role
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
# MIDFIELD ROLES
# =====================================

MIDFIELD_ROLES = [

    "Creative Playmaker",

    "Deep Playmaker",

    "Ball Winner",

    "Box-to-Box"

]


# =====================================
# MIDFIELD TEMPLATES
# =====================================

MIDFIELD_TEMPLATES = {

    "Creative Playmaker": [

        "Ball Winner",
        "Box-to-Box",
        "Deep Playmaker"

    ],

    "Deep Playmaker": [

        "Creative Playmaker",
        "Ball Winner",
        "Box-to-Box"

    ],

    "Ball Winner": [

        "Creative Playmaker",
        "Deep Playmaker",
        "Box-to-Box"

    ],

    "Box-to-Box": [

        "Creative Playmaker",
        "Ball Winner",
        "Deep Playmaker"

    ]

}


# =====================================
# GET MIDFIELDERS
# =====================================

def get_midfield_players():

    players = []

    for idx in range(len(df)):

        pos = str(
            df.iloc[idx]["Position"]
        )

        if pos in [

            "CDM",
            "CM",
            "CAM"

        ]:

            players.append(idx)

    return players


# =====================================
# GET PLAYERS BY ROLE
# =====================================

def get_players_by_role(role):

    players = []

    for idx in get_midfield_players():

        player = df.iloc[idx]

        player_role = get_primary_role(
            player
        )

        if player_role == role:

            players.append(idx)

    return players


# =====================================
# PAIR CHEMISTRY
# =====================================

def pair_score(
    idx1,
    idx2
):

    return compatibility_score(
        idx1,
        idx2
    )


# =====================================
# ROLE FIT
# =====================================

def role_fit(
    anchor_role,
    role
):

    wanted_roles = MIDFIELD_TEMPLATES.get(
        anchor_role,
        []
    )

    if role in wanted_roles:
        return 1.0

    if role == anchor_role:
        return 0.50

    if role in MIDFIELD_ROLES:
        return 0.25

    return 0


# =====================================
# TRIO CHEMISTRY
# =====================================

def trio_chemistry(
    a,
    b,
    c
):

    return (

        pair_score(a, b)

        +

        pair_score(a, c)

        +

        pair_score(b, c)

    ) / 3


# =====================================
# TACTICAL BALANCE
# =====================================

def tactical_balance(
    anchor_idx,
    idx1,
    idx2
):

    anchor_role = get_primary_role(
        df.iloc[anchor_idx]
    )

    role1 = get_primary_role(
        df.iloc[idx1]
    )

    role2 = get_primary_role(
        df.iloc[idx2]
    )

    role_score = (

        role_fit(
            anchor_role,
            role1
        )

        +

        role_fit(
            anchor_role,
            role2
        )

    ) / 2

    chemistry = trio_chemistry(

        anchor_idx,

        idx1,

        idx2

    )

    return (

        0.60 * role_score

        +

        0.40 * chemistry

    )


# =====================================
# BUILD CANDIDATE POOLS
# =====================================

def build_candidate_pools(
    anchor_idx
):

    anchor_role = get_primary_role(
        df.iloc[anchor_idx]
    )

    candidate_pools = {}

    for role in MIDFIELD_ROLES:

        role_players = get_players_by_role(
            role
        )

        scored = []

        for idx in role_players:

            if idx == anchor_idx:
                continue

            chemistry = pair_score(
                anchor_idx,
                idx
            )

            scored.append(
                (
                    idx,
                    chemistry
                )
            )

        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        candidate_pools[role] = [

            idx

            for idx, score

            in scored[:15]

        ]

    return candidate_pools

# =====================================
# MIDFIELD QUALITY BONUS
# =====================================

def midfield_quality_bonus(
    idx
):

    player = df.iloc[idx]

    role = get_primary_role(
        player
    )

    if role == "Creative Playmaker":

        return (

            player["Vision"]

            +

            player["Passing"]

            +

            player["Ball Control"]

            +

            player["Dribbling"]

        ) / 400

    elif role == "Deep Playmaker":

        return (

            player["Vision"]

            +

            player["Passing"]

            +

            player["Ball Control"]

            +

            player["Crossing"]

        ) / 400

    elif role == "Ball Winner":

        return (

            player["Defending"]

            +

            player["Interceptions"]

            +

            player["Standing Tackle"]

            +

            player["Aggression"]

        ) / 400

    elif role == "Box-to-Box":

        return (

            player["Physicality"]

            +

            player["Pace"]

            +

            player["Passing"]

            +

            player["Dribbling"]

        ) / 400

    return 0


# =====================================
# FIND BEST TRIO
# =====================================

def best_trio(
    anchor_idx
):

    anchor_role = get_primary_role(
        df.iloc[anchor_idx]
    )

    if anchor_role not in MIDFIELD_ROLES:

        return None, 0

    candidate_pools = build_candidate_pools(
        anchor_idx
    )

    best_score = -1

    best_trio_result = None

    for role_a in MIDFIELD_ROLES:

        for role_b in MIDFIELD_ROLES:

            if role_a == role_b:
                continue

            pool_a = candidate_pools[
                role_a
            ]

            pool_b = candidate_pools[
                role_b
            ]

            for idx_a in pool_a:

                for idx_b in pool_b:

                    if idx_a == idx_b:
                        continue

                    if idx_a == anchor_idx:
                        continue

                    if idx_b == anchor_idx:
                        continue

                    tactical = tactical_balance(

                        anchor_idx,

                        idx_a,

                        idx_b

                    )

                    quality = (

                        midfield_quality_bonus(
                            idx_a
                        )

                        +

                        midfield_quality_bonus(
                            idx_b
                        )

                    ) / 2

                    role_a_fit = role_fit(

                        anchor_role,

                        get_primary_role(
                            df.iloc[idx_a]
                        )

                    )

                    role_b_fit = role_fit(

                        anchor_role,

                        get_primary_role(
                            df.iloc[idx_b]
                        )

                    )

                    final_score = (

                        0.50 * tactical

                        +

                        0.20 * quality

                        +

                        0.15 * role_a_fit

                        +

                        0.15 * role_b_fit

                    )

                    if final_score > best_score:

                        best_score = final_score

                        best_trio_result = (

                            anchor_idx,

                            idx_a,

                            idx_b

                        )

    return (

        best_trio_result,

        best_score

    )


# =====================================
# PRINT TRIO
# =====================================

def print_trio(
    trio,
    score
):

    if trio is None:

        print(
            "No valid trio found."
        )

        return

    print(
        "\nBEST TACTICAL MIDFIELD\n"
    )

    for idx in trio:

        player = df.iloc[idx]

        role = get_primary_role(
            player
        )

        print(

            f"{player['Player']} "

            f"| {player['Position']} "

            f"| {role}"

        )

    print(
        f"\nTactical Score: "
        f"{score:.3f}"
    )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    player_idx = find_player(
        player_name
    )

    if player_idx is None:

        print(
            "Player not found."
        )

        exit()

    trio, score = best_trio(
        player_idx
    )

    print_trio(
        trio,
        score
    )