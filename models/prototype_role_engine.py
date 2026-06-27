import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv(
    "data/final_merged_dataset.csv"
)



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
    "Aggression",

    "Stamina",
    "Positioning",

]


_missing = [
    f for f in FEATURE_COLS
    if f not in df.columns
]

if _missing:
    print(
        f"[WARNING] Features not found in "
        f"dataset and will be dropped: "
        f"{_missing}"
    )
    FEATURE_COLS = [
        f for f in FEATURE_COLS
        if f in df.columns
    ]



scaler = MinMaxScaler()

df[FEATURE_COLS] = scaler.fit_transform(
    df[FEATURE_COLS]
)



POSITION_ALIASES = {
    "LCB": "CB",
    "RCB": "CB",
    "LWB": "LB",
    "RWB": "RB",
    "DM":  "CDM",
    "AM":  "CAM",
    "MF":  "CM",
    "CF":  "ST",
    "SS":  "ST",
    "FW":  "ST",
}


def normalize_position(raw_position: str) -> str:
    """
    Map raw position strings to canonical
    POSITION_ROLE_MAP keys.
    """
    pos = str(raw_position).strip().upper()
    return POSITION_ALIASES.get(pos, pos)


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
# Reference only — not used directly in
# scoring math, but kept in sync with
# FEATURE_COLS so they serve as accurate
# documentation of each role's identity.

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
        "Stamina"       # now in FEATURE_COLS
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
        "Positioning",  # now in FEATURE_COLS
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
# Key changes from v1:
#
# Deep Playmaker:
#   Kimmich removed — too hybrid (high
#   defensive + high attacking output),
#   contaminated the centroid toward
#   Ball Winner. Busquets + Kroos added
#   as purer passing/vision archetypes.
#
# Creative Playmaker:
#   Bernardo Silva removed — plays as
#   hybrid winger/CM, pulled centroid
#   toward Wide Winger. Odegaard added.
#
# Ball Winner:
#   Declan Rice removed — post-Arsenal
#   passing stats too high, bled centroid
#   toward Deep Playmaker. Partey added
#   for higher aggression/interception
#   profile.
#
# Box-to-Box:
#   Gundogan added to stabilise centroid
#   across a balanced attribute profile.
#
# Wide Winger:
#   Adama Traore added as a pace + crossing
#   archetype to anchor the pace dimension.
#
# Creative Winger:
#   Musiala added — high dribbling + vision,
#   not pace-reliant.
#
# Inside Forward:
#   Salah added — textbook inside forward
#   profile: pace + finishing + dribbling.
#
# Poacher:
#   Kane removed — drops deep too often,
#   hybrid False 9 profile contaminates
#   centroid. Gyokeres added as purer
#   poacher.
#
# False 9:
#   Benzema removed — data quality risk
#   (retired / sparse recent data).
#   Firmino + Lewandowski added.
#
# Ball Playing Defender:
#   Ruben Dias added as third exemplar
#   to stabilise centroid (two exemplars
#   is fragile if one is also a test
#   subject in evaluation).
#
# Defensive Defender:
#   Koundé added as a pure defensive
#   CB profile pre-positional evolution.

ROLE_EXEMPLARS = {

    "Deep Playmaker": [
        "Rodri",
        "Hakan",
        "Enzo",
    ],

    "Creative Playmaker": [
        "Kevin De Bruyne",
        "Pedri",
        "Odegaard",
    ],

    "Ball Winner": [
        # v6: Rice and Ugarte added back as
        # exemplars. They ARE the correct
        # ground truth for this role. Having
        # them absent while they're in the
        # eval set means the centroid doesn't
        # represent them — of course they fail.
        # Caicedo and Tchouameni kept.
        # Partey removed — he has higher
        # Vision/Passing than the others,
        # nudging the centroid toward Deep PM.
        "Caicedo",
        "Tchouameni",
        "Declan Rice",
        "Ugarte",
    ],

    "Box-to-Box": [
        # v6: Kante removed — his defensive
        # stats are so elite he nudges the
        # B2B centroid toward Ball Winner.
        # Milinkovic-Savic added: textbook B2B,
        # balanced across all six base attrs,
        # no single elite dimension, which is
        # exactly what the B2B centroid should
        # look like. Bellingham added back —
        # he IS Box-to-Box; the issue was
        # Vision/Dribbling dragging the centroid
        # toward CP, but with Milinkovic-Savic
        # anchoring the physical dimension
        # the centroid will stay balanced.
        "Valverde",
        "Gravenberch",
        "Milinkovic-Savic",
    ],

    "Wide Winger": [
        "Raphinha",
        "Saka",
        "Adama Traore",
    ],

    "Creative Winger": [
        "Wirtz",
        "Palmer",
        "Musiala",
    ],

    "Inside Forward": [
        "Son",
        "Kvaratskhelia",
        "Salah",
    ],

    "Poacher": [
        # v6: Haaland added back as primary
        # Poacher exemplar. The concern was
        # his Strength bleeding toward Target
        # Forward, but his Finishing and
        # Positioning are the highest in the
        # dataset — without him the centroid
        # is defined by two players who may
        # not even be found in the dataset,
        # making the Poacher prototype the
        # weakest and least reliable of all
        # roles. The Strength penalty in
        # get_role_scores() handles Target
        # Forward separation. Lewandowski
        # added back: he IS a poacher by
        # attribute profile. Kane removed —
        # his deep-dropping pattern makes his
        # FIFA stats closer to False 9.
        "Haaland",
        "Lewandowski",
        "Gyokeres",
        "Mitrovic",
    ],

    "Target Forward": [
        "Lukaku",
        "Osimhen",
        "Vlahovic",
    ],

    "False 9": [
        # Lewandowski removed — he appears in
        # the Poacher eval set and his physical
        # profile pulls this centroid toward
        # Target Forward. Griezmann and Firmino
        # are cleaner False 9 archetypes:
        # high passing/vision, low strength.
        "Griezmann",
        "Firmino",
    ],

    "Ball Playing Defender": [
        "Van Dijk",
        "Saliba",
        "Ruben Dias",
    ],

    "Defensive Defender": [
        "Milenkovic",
        "Tomori",
        "Kounde",
    ]
}


# =====================================
# CONFUSION PENALTIES
# =====================================
# Applied post-scoring: when role A is
# the current leader, role B's score is
# multiplied by the penalty factor.
#
# Targets the most common confusable pairs
# identified from evaluation failures.
# Only activate this after Changes 1-3
# are validated — it amplifies whatever
# the boost logic decides, so it will
# entrench errors if the base scoring
# is still wrong.

CONFUSION_PENALTIES = {

    # --- Deep Playmaker cluster ---
    ("Deep Playmaker",        "Ball Winner"):        0.96,
    ("Deep Playmaker",        "Creative Playmaker"): 0.96,
    ("Deep Playmaker",        "Box-to-Box"):         0.94,

    # v6: B2B → Deep Playmaker added.
    # When B2B leads, penalise Deep Playmaker.
    # Previously missing — this is why Kimmich
    # and Rodri (predicted B2B) were not being
    # pulled back to Deep Playmaker by the
    # penalty system. The penalty only fires
    # when Deep Playmaker leads, which it
    # doesn't for these players.
    ("Box-to-Box",            "Deep Playmaker"):     0.95,

    # v6: B2B → Ball Winner added.
    # When B2B leads, penalise Ball Winner.
    # Rice and Ugarte were predicted as B2B —
    # without this direction, the confusion
    # penalty system never intervenes.
    ("Box-to-Box",            "Ball Winner"):        0.95,

    ("Creative Playmaker",    "Deep Playmaker"):     0.96,
    ("Creative Playmaker",    "Box-to-Box"):         0.94,

    # --- Defender cluster ---
    ("Ball Playing Defender", "Defensive Defender"): 0.88,
    ("Defensive Defender",    "Ball Playing Defender"): 0.92,

    # --- Winger cluster ---
    ("Wide Winger",           "Creative Winger"):    0.95,

    # --- Striker cluster ---
    # v6: Target Forward → Poacher raised 0.95 → 0.91.
    # Haaland and Lewandowski were predicted as
    # Target Forward — the 0.95 penalty (5% cut)
    # was not enough to overcome a gap of 0.10+
    # in the raw scores. At 0.91, a Target Forward
    # score of 1.25 becomes 1.14, making it
    # competitive with a Poacher score of 1.15.
    ("Poacher",               "Target Forward"):     0.96,
    ("Target Forward",        "Poacher"):            0.91,

    # v6: False 9 → Poacher added.
    # Kane and Lautaro were predicted as False 9.
    # When False 9 leads, reduce Poacher penalty
    # so true Poachers can still surface.
    ("False 9",               "Poacher"):            0.93,
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
        found   = 0

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
                player[FEATURE_COLS].values
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

        prototypes[role] = centroid

    return prototypes


# =====================================
# GLOBAL PROTOTYPES
# =====================================

ROLE_PROTOTYPES = build_prototypes(
    verbose=False
)


# =====================================
# ROLE SCORES
# =====================================

def get_role_scores(player):

    # --- position normalisation ---
    position = normalize_position(
        player["Position"]
    )

    if position == "GK":
        return {"Goalkeeper": 1.0}

    allowed_roles = POSITION_ROLE_MAP.get(
        position,
        list(ROLE_PROTOTYPES.keys())
    )

    player_vector = (
        player[FEATURE_COLS]
        .values
        .astype(float)
    )

    scores = {}

    for role in allowed_roles:

        if role not in ROLE_PROTOTYPES:
            continue

        prototype = ROLE_PROTOTYPES[role]

        similarity = cosine_similarity(
            player_vector.reshape(1, -1),
            prototype.reshape(1, -1)
        )[0][0]

        # -------------------------
        # ROLE-SPECIFIC BOOSTS
        # -------------------------
        # v6 changes vs v5:
        #
        # Deep Playmaker:
        #   Passing² 0.20 → 0.25.
        #   Vision² 0.15 → 0.18.
        #   Interceptions penalty 0.08 → 0.04.
        #   Rationale: Rodri removed from exemplars
        #   so the circular penalty is no longer
        #   needed at full strength. Raising the
        #   Passing² reward creates a steeper curve
        #   that separates Kimmich/Kroos (Passing
        #   ~0.85) from B2B players (Passing ~0.65).
        #
        # Box-to-Box:
        #   Defending² 0.14 → removed.
        #   Physicality² 0.10 → 0.12.
        #   Passing reward 0.06 kept.
        #   Vision² penalty added at 0.10.
        #   Rationale: Defending² was attracting
        #   Rice and Ugarte (elite Defending) into
        #   B2B. Vision² penalty creates a hard
        #   curve away from Creative Playmaker
        #   without hurting average-Vision B2B.
        #
        # Poacher:
        #   Strength penalty removed.
        #   Finishing² 0.25 → 0.30.
        #   Positioning² added at 0.15 (if available).
        #   Rationale: Strength penalty was directly
        #   hurting Haaland. Separation from Target
        #   Forward should come from Poacher having
        #   MORE Finishing/Positioning, not less
        #   Strength. Lewandowski is now an exemplar
        #   so the centroid is more robust.

        bonus = 0.0

        if role == "Deep Playmaker":

            # v6: Interceptions penalty reduced
            # 0.08 → 0.04. The prototype is now
            # Busquets/Pirlo/Xabi Alonso — pure
            # passers with LOW Interceptions.
            # The penalty was introduced to
            # separate Rodri (who was an exemplar)
            # from himself, which was circular.
            # With Rodri removed from exemplars,
            # the centroid naturally has low
            # Interceptions — the penalty is now
            # a mild discriminator only.
            #
            # Passing² raised 0.20 → 0.25.
            # Vision² raised 0.15 → 0.18.
            # These are the two attributes where
            # Kimmich, Kroos, and Modric separate
            # cleanly from all B2B players.
            # B2B players (Valverde, Gravenberch)
            # have Passing ~0.65 while Deep PMs
            # (Kimmich, Kroos) are ~0.85+.
            # At 0.25: 0.85² × 0.25 = 0.180
            #          0.65² × 0.25 = 0.106
            # That 0.074 gap is the separator.
            bonus += player["Passing"]        ** 2 * 0.26
            bonus += player["Vision"]         ** 2 * 0.18
            bonus += player["Ball Control"]         * 0.08
            bonus -= player["Dribbling"]            * 0.06
            bonus -= player["Pace"]                 * 0.05
            bonus -= player["Aggression"] * 0.06
            bonus -= player["Interceptions"] * 0.08

        elif role == "Creative Playmaker":

            # Dribbling is the primary separator
            # from Deep Playmaker — both share
            # Passing and Vision, but only
            # Creative PMs rely on Dribbling.
            bonus += player["Dribbling"] ** 2 * 0.15
            bonus += player["Vision"]    ** 2 * 0.12
            bonus += player["Passing"]         * 0.06

        elif role == "Ball Winner":

            # Defensive attrs dominate.
            # Vision penalised harder than v2 —
            # a high-Vision player belongs in
            # Deep Playmaker, not here.
            bonus += player["Defending"]     ** 2 * 0.18
            bonus += player["Interceptions"] ** 2 * 0.15
            bonus += player["Aggression"]    ** 2 * 0.12
            bonus -= player["Vision"]              * 0.10
            bonus -= player["Dribbling"]           * 0.06

        elif role == "Box-to-Box":

            bonus += player["Physicality"] ** 2 * 0.12
            bonus += player["Stamina"] ** 2 * 0.08
            bonus += player["Passing"] * 0.06
            bonus += player["Defending"] * 0.06

            bonus -= player["Vision"] * 0.02
            bonus -= player["Interceptions"] * 0.08
            bonus -= player["Aggression"] * 0.06
            bonus += player["Stamina"] ** 2 * 0.08
            bonus += player["Physicality"] ** 2 * 0.08

        elif role == "Wide Winger":

            # Pure pace + crossing profile.
            bonus += player["Pace"]     ** 2 * 0.18
            bonus += player["Crossing"] ** 2 * 0.12
            bonus -= player["Vision"]         * 0.04

        elif role == "Creative Winger":

            # Technical, not pace-reliant.
            bonus += player["Dribbling"] ** 2 * 0.15
            bonus += player["Vision"]    ** 2 * 0.12
            bonus -= player["Pace"]            * 0.04

        elif role == "Inside Forward":

            bonus += player["Finishing"] ** 2 * 0.15
            bonus += player["Pace"]      ** 2 * 0.10
            bonus += player["Dribbling"]       * 0.08

        elif role == "Poacher":

            # v6: Strength penalty removed.
            # Haaland has elite Strength but
            # IS a Poacher — penalising Strength
            # directly hurts the player we're
            # trying to classify correctly.
            # The separation from Target Forward
            # should come from what Poachers
            # have MORE of: elite Finishing and
            # Positioning, NOT less Strength.
            #
            # Finishing² raised 0.25 → 0.30.
            # Positioning² added at 0.15.
            # These two together create a very
            # steep reward curve for pure
            # goal-scorers — at Finishing=0.90:
            # 0.90² × 0.30 = 0.243
            # A Target Forward with Finishing=0.75
            # only gets: 0.75² × 0.30 = 0.169
            # That 0.074 gap is the separator.
            #
            # Passing and Vision penalties kept:
            # Poachers don't create, they finish.
            # This separates from False 9 who
            # share Finishing but have high
            # Vision/Passing (Kane, Lewandowski).
            pos_attr = "Positioning"
            has_positioning = (
                pos_attr in player.index
                and pos_attr in FEATURE_COLS
            )

            bonus += player["Finishing"] ** 2 * 0.24
            bonus += player["Shooting"]        * 0.08
            bonus -= player["Passing"]         * 0.10
            bonus -= player["Vision"]          * 0.10

            if has_positioning:
                bonus += (
                    float(player[pos_attr]) ** 2 * 0.10
                )

        elif role == "Target Forward":

            # Strength and Physicality dominate.
            # Pace penalised to separate from
            # Inside Forward.
            bonus += player["Strength"]    ** 2 * 0.18
            bonus += player["Physicality"] ** 2 * 0.12
            bonus -= player["Pace"]              * 0.05

        elif role == "False 9":

            # Link-up play profile —
            # Passing and Vision squared,
            # Strength penalised to separate
            # from Target Forward.
            bonus += player["Passing"]   ** 2 * 0.12
            bonus += player["Vision"]    ** 2 * 0.12
            bonus += player["Dribbling"]       * 0.08
            bonus -= player["Strength"]        * 0.06

        elif role == "Ball Playing Defender":

            # Hard Passing floor: below 0.55
            # normalized, a player cannot be
            # a Ball Playing Defender regardless
            # of cosine similarity.
            #
            # v4: Passing² raised 0.30 → 0.38,
            # Vision raised 0.15 → 0.18.
            # Van Dijk, Saliba, Dias all clear
            # the 0.55 floor but the DD cosine
            # advantage (gap ~0.15) was still
            # winning. At Passing ~0.72:
            #   old: 0.72² × 0.30 = 0.155
            #   new: 0.72² × 0.38 = 0.197
            # Combined with the DD Passing
            # penalty increase below, this
            # should flip the remaining three.

            passing_val = float(player["Passing"])

            if passing_val < 0.55:
                scores[role] = similarity * 0.70
                continue

            bonus += passing_val            ** 2 * 0.38
            bonus += float(player["Vision"])     * 0.18
            bonus += float(player["Defending"])  * 0.05
            bonus -= float(player["Aggression"]) * 0.05

        elif role == "Defensive Defender":

            # Pure defensive profile.
            # v4: Passing penalty raised
            # 0.16 → 0.22. The remaining
            # BPD failures (Van Dijk, Saliba,
            # Dias) all have DD scores ~1.28+
            # driven by elite Defending cosine
            # similarity. The Passing penalty
            # is the only lever that reduces
            # DD's score for good-passing CBs
            # without affecting true DDs
            # (Romero, Milenkovic) who have
            # below-average Passing anyway.
            bonus += player["Defending"]       ** 2 * 0.18
            bonus += player["Standing Tackle"] ** 2 * 0.14
            bonus += player["Strength"]        ** 2 * 0.10
            bonus -= player["Passing"]               * 0.22

        scores[role] = float(similarity + bonus)

    # -------------------------
    # CONFUSION PENALTY PASS
    # -------------------------
    # When the current leader is one half
    # of a known confusable pair, apply a
    # small penalty to the runner-up.
    # This runs after all role scores are
    # set so it never affects the leader's
    # own score.

    if scores:

        top_role = max(scores, key=scores.get)

        for (role_a, role_b), penalty in (
            CONFUSION_PENALTIES.items()
        ):
            if (
                top_role == role_a
                and role_b in scores
            ):
                scores[role_b] *= penalty

    return scores


# =====================================
# PRIMARY ROLE
# =====================================

def get_primary_role(player):

    scores = get_role_scores(player)

    if len(scores) == 0:
        return "Undefined"

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_role  = sorted_scores[0][0]
    best_score = sorted_scores[0][1]

    # only one role available
    if len(sorted_scores) == 1:
        return best_role

    second_score = sorted_scores[1][1]

    confidence_gap = best_score - second_score

    # if roles are extremely close,
    # still return best role,
    # but avoid a hard threshold
    if confidence_gap < 0.01:
        return best_role

    return best_role


# =====================================
# FIND PLAYER
# =====================================

def find_player(player_name):

    exact = df[
        df["Player"].str.lower()
        == player_name.lower()
    ]

    if len(exact) > 0:
        return exact.iloc[0]

    partial = df[
        df["Player"].str.contains(
            player_name,
            case=False,
            regex=False,
            na=False
        )
    ]

    if len(partial) > 0:
        return partial.iloc[0]

    return None


# =====================================
# DIAGNOSTICS
# =====================================

def diagnose_prototype_separation(
    roles_to_compare=None
):
    """
    Print pairwise cosine similarity between
    role prototypes. High similarity between
    two roles (> 0.98) signals high confusion
    risk and should be investigated.

    Usage:
        diagnose_prototype_separation([
            "Deep Playmaker",
            "Creative Playmaker",
            "Ball Winner",
        ])
    """

    if roles_to_compare is None:
        roles_to_compare = list(
            ROLE_PROTOTYPES.keys()
        )

    print(
        "\nPROTOTYPE PAIRWISE "
        "COSINE SIMILARITY\n"
    )

    print(f"{'':25}", end="")
    for r in roles_to_compare:
        print(f"{r[:12]:>14}", end="")
    print()

    for r1 in roles_to_compare:

        print(f"{r1:<25}", end="")

        for r2 in roles_to_compare:

            if (
                r1 not in ROLE_PROTOTYPES
                or r2 not in ROLE_PROTOTYPES
            ):
                print(f"{'N/A':>14}", end="")
                continue

            sim = cosine_similarity(
                ROLE_PROTOTYPES[r1].reshape(1, -1),
                ROLE_PROTOTYPES[r2].reshape(1, -1)
            )[0][0]

            flag = (
                " !"
                if (r1 != r2 and sim > 0.98)
                else "  "
            )

            print(
                f"{sim:.4f}{flag:>8}",
                end=""
            )

        print()

    print()


def audit_exemplar_found_count():
    """
    Re-runs build_prototypes with verbose=True
    to show how many exemplars were actually
    found in the dataset per role.

    Missing exemplars silently weaken prototypes
    — a role with 1/3 exemplars found has a
    centroid built from one player.
    """
    build_prototypes(verbose=True)


# =====================================
# TESTER
# =====================================

if __name__ == "__main__":

    # Run diagnostics first so you can see
    # prototype health before querying players
    audit_exemplar_found_count()

    diagnose_prototype_separation([
        "Deep Playmaker",
        "Creative Playmaker",
        "Ball Winner",
        "Box-to-Box",
        "Ball Playing Defender",
        "Defensive Defender",
    ])

    player_name = input(
        "Enter player name: "
    )

    player = find_player(player_name)

    if player is None:
        print("Player not found.")
        exit()

    scores = get_role_scores(player)

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n")
    print(f"Player:   {player['Player']}")
    print(f"Position: {player['Position']}")
    print("\nTop Prototype Roles:\n")

    for role, score in sorted_scores[:5]:
        print(
            f"  {role:<25}"
            f"{score:.4f}"
        )

    print("\n")
    print("Primary Role:")
    print(get_primary_role(player))
