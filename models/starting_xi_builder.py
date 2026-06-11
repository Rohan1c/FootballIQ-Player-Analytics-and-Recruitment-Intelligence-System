import pandas as pd

from models.compatibility_engine import (
    compatibility_score,
    df
)

from models.prototype_role_engine import (
    get_primary_role,
    get_role_scores
)

# =====================================
# POSITION MAP
# =====================================

POSITION_MAP = {

    "GK": ["GK"],

    "RB": ["RB"],

    "LB": ["LB"],

    "CB": ["CB"],

    "CDM": ["CDM"],

    "CM": [
        "CM",
        "CAM"
    ],

    "RW": [
        "RW",
        "RM"
    ],

    "LW": [
        "LW",
        "LM"
    ],

    "ST": [
        "ST"
    ]
}


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
# POSITION FILTER
# =====================================

def get_players_for_position(
    tactical_position
):

    valid_positions = POSITION_MAP[
        tactical_position
    ]

    candidates = []

    for idx in range(len(df)):

        fifa_position = str(
            df.iloc[idx]["Position"]
        )

        if fifa_position in valid_positions:

            candidates.append(idx)

    return candidates


# =====================================
# ROLE BONUS
# =====================================

def role_bonus(
    player_idx,
    tactical_position
):

    player = df.iloc[
        player_idx
    ]

    role = get_primary_role(
        player
    )

    if tactical_position == "CDM":

        if role == "Ball Winner":
            return 0.10

        if role == "Deep Playmaker":
            return 0.08

    elif tactical_position == "CM":

        if role == "Creative Playmaker":
            return 0.10

        if role == "Box-to-Box":
            return 0.08

    elif tactical_position == "RW":

        if role == "Wide Winger":
            return 0.10

        if role == "Creative Winger":
            return 0.08

    elif tactical_position == "LW":

        if role == "Inside Forward":
            return 0.10

        if role == "Creative Winger":
            return 0.08

    elif tactical_position == "ST":

        if role == "Target Forward":
            return 0.10

        if role == "Poacher":
            return 0.08

        if role == "False 9":
            return 0.08

    elif tactical_position == "CB":

        if role == "Defensive Defender":
            return 0.10

        if role == "Ball Playing Defender":
            return 0.08

    return 0


# =====================================
# PLAYER SCORE
# =====================================

def candidate_score(

    anchor_idx,

    candidate_idx,

    tactical_position

):

    chemistry = compatibility_score(

        anchor_idx,

        candidate_idx

    )

    bonus = role_bonus(

        candidate_idx,

        tactical_position

    )

    return (

        chemistry

        +

        bonus

    )


# =====================================
# PICK PLAYER
# =====================================

def pick_best_player(

    anchor_idx,

    tactical_position,

    used_players

):

    candidates = get_players_for_position(
        tactical_position
    )

    best_idx = None

    best_score = -999

    for idx in candidates:

        if idx == anchor_idx:
            continue

        if idx in used_players:
            continue

        score = candidate_score(

            anchor_idx,

            idx,

            tactical_position

        )

        if score > best_score:

            best_score = score

            best_idx = idx

    return best_idx

# =====================================
# BUILD XI
# =====================================

def build_xi(anchor_idx):

    used_players = {
        anchor_idx
    }

    xi = {}

    anchor_player = df.iloc[
        anchor_idx
    ]

    anchor_position = str(
        anchor_player["Position"]
    )

    # -------------------------
    # MIDFIELD
    # -------------------------

    if anchor_position == "CDM":

        xi["CDM"] = anchor_idx

        cm1 = pick_best_player(
            anchor_idx,
            "CM",
            used_players
        )

        used_players.add(cm1)

        cm2 = pick_best_player(
            anchor_idx,
            "CM",
            used_players
        )

        used_players.add(cm2)

        xi["CM1"] = cm1
        xi["CM2"] = cm2

    elif anchor_position in [

        "CM",
        "CAM"

    ]:

        xi["CM1"] = anchor_idx

        cdm = pick_best_player(
            anchor_idx,
            "CDM",
            used_players
        )

        used_players.add(cdm)

        cm2 = pick_best_player(
            anchor_idx,
            "CM",
            used_players
        )

        used_players.add(cm2)

        xi["CDM"] = cdm
        xi["CM2"] = cm2

    else:

        cdm = pick_best_player(
            anchor_idx,
            "CDM",
            used_players
        )

        used_players.add(cdm)

        cm1 = pick_best_player(
            anchor_idx,
            "CM",
            used_players
        )

        used_players.add(cm1)

        cm2 = pick_best_player(
            anchor_idx,
            "CM",
            used_players
        )

        used_players.add(cm2)

        xi["CDM"] = cdm
        xi["CM1"] = cm1
        xi["CM2"] = cm2

    # -------------------------
    # DEFENCE
    # -------------------------

    gk = pick_best_player(
        anchor_idx,
        "GK",
        used_players
    )

    used_players.add(gk)

    rb = pick_best_player(
        anchor_idx,
        "RB",
        used_players
    )

    used_players.add(rb)

    cb1 = pick_best_player(
        anchor_idx,
        "CB",
        used_players
    )

    used_players.add(cb1)

    cb2 = pick_best_player(
        anchor_idx,
        "CB",
        used_players
    )

    used_players.add(cb2)

    lb = pick_best_player(
        anchor_idx,
        "LB",
        used_players
    )

    used_players.add(lb)

    xi["GK"] = gk
    xi["RB"] = rb
    xi["CB1"] = cb1
    xi["CB2"] = cb2
    xi["LB"] = lb

    # -------------------------
    # ATTACK
    # -------------------------

    rw = pick_best_player(
        anchor_idx,
        "RW",
        used_players
    )

    used_players.add(rw)

    st = pick_best_player(
        anchor_idx,
        "ST",
        used_players
    )

    used_players.add(st)

    lw = pick_best_player(
        anchor_idx,
        "LW",
        used_players
    )

    used_players.add(lw)

    xi["RW"] = rw
    xi["ST"] = st
    xi["LW"] = lw

    return xi


# =====================================
# PRINT XI
# =====================================

def print_xi(xi):

    print("\n")
    print("=" * 70)
    print("RECOMMENDED 4-3-3")
    print("=" * 70)

    order = [

        "GK",

        "RB",
        "CB1",
        "CB2",
        "LB",

        "CDM",
        "CM1",
        "CM2",

        "RW",
        "ST",
        "LW"

    ]

    for pos in order:

        idx = xi[pos]

        player = df.iloc[idx]

        role = get_primary_role(
            player
        )

        fifa_position = str(
            player["Position"]
        )

        print(

            f"{pos:<5}"

            f"{player['Player']:<30}"

            f"{fifa_position:<6}"

            f"{role}"

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

    xi = build_xi(
        player_idx
    )

    print_xi(
        xi
    )