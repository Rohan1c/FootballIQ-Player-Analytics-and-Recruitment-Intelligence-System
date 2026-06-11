import pandas as pd
import numpy as np

from models.prototype_role_engine import (
    df,
    get_primary_role,
    get_role_scores
)

from models.compatibility_engine import (
    compatibility_score,
    find_player
)

from models.transfer_engine import (
    replacement_score
)

from models.squad_builder import (
    best_trio
)

from models.starting_xi_builder import (
    build_xi
)

# =====================================
# TOP ATTRIBUTES
# =====================================

ATTRIBUTE_COLUMNS = [

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


def top_attributes(

    player,

    top_n=5

):

    scores = []

    for col in ATTRIBUTE_COLUMNS:

        try:

            scores.append(

                (

                    col,

                    float(
                        player[col]
                    )

                )

            )

        except:

            pass

    scores.sort(

        key=lambda x: x[1],

        reverse=True

    )

    return scores[:top_n]


# =====================================
# BEST PARTNERS
# =====================================

def get_best_partners(

    player_idx,

    top_n=5

):

    scores = []

    for idx in range(len(df)):

        if idx == player_idx:
            continue

        score = compatibility_score(

            player_idx,

            idx

        )

        scores.append(

            (

                idx,

                score

            )

        )

    scores.sort(

        key=lambda x: x[1],

        reverse=True

    )

    return scores[:top_n]


# =====================================
# BEST REPLACEMENTS
# =====================================

def get_best_replacements(

    player_idx,

    top_n=5

):

    scores = []

    for idx in range(len(df)):

        if idx == player_idx:
            continue

        score = replacement_score(

            player_idx,

            idx

        )

        scores.append(

            (

                idx,

                score

            )

        )

    scores.sort(

        key=lambda x: x[1],

        reverse=True

    )

    return scores[:top_n]


# =====================================
# REPORT
# =====================================

def generate_report(

    player_name

):

    player_idx = find_player(
        player_name
    )

    if player_idx is None:

        print(
            "Player not found."
        )

        return

    player = df.iloc[
        player_idx
    ]

    role = get_primary_role(
        player
    )

    role_scores = (
        get_role_scores(
            player
        )
    )

    sorted_roles = sorted(

        role_scores.items(),

        key=lambda x: x[1],

        reverse=True

    )

    print("\n")
    print("=" * 80)
    print(
        "FOOTBALL INTELLIGENCE REPORT"
    )
    print("=" * 80)

    print("\nPLAYER PROFILE\n")

    print(
        f"Name: "
        f"{player['Player']}"
    )

    print(
        f"Position: "
        f"{player['Position']}"
    )

    print(
        f"Age: "
        f"{player['Age_y']}"
    )

    print(
        f"Primary Role: "
        f"{role}"
    )

    print("\nTOP ROLE SCORES\n")

    for role_name, score in sorted_roles[:5]:

        print(

            f"{role_name:<25}"

            f"{score:.3f}"

        )

    print("\nTOP ATTRIBUTES\n")

    for attr, value in top_attributes(
        player
    ):

        print(

            f"{attr:<25}"

            f"{value:.2f}"

        )
        # =====================================
    # BEST PARTNERS
    # =====================================

    partners = get_best_partners(
        player_idx,
        top_n=5
    )

    print("\n")
    print("=" * 80)
    print("BEST COMPATIBLE PARTNERS")
    print("=" * 80)

    for rank, (idx, score) in enumerate(
        partners,
        start=1
    ):

        partner = df.iloc[idx]

        print(

            f"{rank}. "

            f"{partner['Player']} "

            f"| {partner['Position']} "

            f"| {get_primary_role(partner)} "

            f"| {score:.3f}"

        )

    # =====================================
    # BEST REPLACEMENTS
    # =====================================

    replacements = get_best_replacements(
        player_idx,
        top_n=5
    )

    print("\n")
    print("=" * 80)
    print("BEST REPLACEMENTS")
    print("=" * 80)

    for rank, (idx, score) in enumerate(
        replacements,
        start=1
    ):

        replacement = df.iloc[idx]

        print(

            f"{rank}. "

            f"{replacement['Player']} "

            f"| {replacement['Position']} "

            f"| {get_primary_role(replacement)} "

            f"| {score:.3f}"

        )

    # =====================================
    # BEST MIDFIELD TRIO
    # =====================================

    print("\n")
    print("=" * 80)
    print("BEST MIDFIELD TRIO")
    print("=" * 80)

    trio, trio_score = best_trio(
        player_idx
    )

    if trio is not None:

        for idx in trio:

            trio_player = df.iloc[idx]

            print(

                f"{trio_player['Player']} "

                f"| {trio_player['Position']} "

                f"| {get_primary_role(trio_player)}"

            )

        print(
            f"\nTactical Score: "
            f"{trio_score:.3f}"
        )

    else:

        print(
            "No midfield trio available."
        )

    # =====================================
    # RECOMMENDED XI
    # =====================================

    print("\n")
    print("=" * 80)
    print("RECOMMENDED STARTING XI")
    print("=" * 80)

    xi = build_xi(
        player_idx
    )

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

        xi_player = df.iloc[idx]

        print(

            f"{pos:<5}"

            f"{xi_player['Player']:<30}"

            f"{xi_player['Position']:<6}"

            f"{get_primary_role(xi_player)}"

        )

    # =====================================
    # PLAYER SUMMARY
    # =====================================

    print("\n")
    print("=" * 80)
    print("FOOTBALL INTELLIGENCE SUMMARY")
    print("=" * 80)

    print(
        f"\n{player['Player']} "
        f"is primarily classified as a "
        f"{role} operating from "
        f"{player['Position']}."
    )

    best_attr = top_attributes(
        player,
        top_n=3
    )

    print(
        "\nKey Strengths:"
    )

    for attr, value in best_attr:

        print(
            f"- {attr}"
        )

    print(
        "\nRecommended Usage:"
    )

    if role == "Creative Playmaker":

        print(
            "- Use as primary creator."
        )

    elif role == "Deep Playmaker":

        print(
            "- Use as tempo controller."
        )

    elif role == "Ball Winner":

        print(
            "- Use as defensive shield."
        )

    elif role == "Box-to-Box":

        print(
            "- Use as transition midfielder."
        )

    elif role == "Inside Forward":

        print(
            "- Use as goalscoring wide attacker."
        )

    elif role == "Poacher":

        print(
            "- Use as penalty-box finisher."
        )

    elif role == "Target Forward":

        print(
            "- Use as focal point striker."
        )

    elif role == "Ball Playing Defender":

        print(
            "- Use to progress possession from defence."
        )

    elif role == "Defensive Defender":

        print(
            "- Use as defensive stopper."
        )

    print("\n")
    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    generate_report(
        player_name
    )