import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("data/final_merged_dataset.csv")

archetypes = pd.read_csv(
    "data/player_archetypes.csv"
)

embeddings = np.load(
    "models/latent_embeddings.npy"
)

distance_matrix = euclidean_distances(
    embeddings
)

max_dist = np.max(distance_matrix)

similarity_matrix = (
    1 - distance_matrix / max_dist
)

ARCHETYPE_NAMES = {
    0: "Balanced Midfielder",
    1: "Creative Attacker",
    2: "Supporting Attacker",
    3: "Elite Outlier",
    4: "Physical Finisher",
    5: "Ball Winner",
    6: "Defender",
    7: "Elite Forward"
}


def get_player_index(player_name):

    exact = df[
        df["Player"].str.lower()
        ==
        player_name.lower()
    ]

    if len(exact) > 0:
        return exact.index[0]

    matches = df[
        df["Player"].str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(matches) > 0:
        return matches.index[0]

    return None


def get_position_fit(pos1, pos2):

    pos1 = str(pos1)
    pos2 = str(pos2)

    if pos1 == pos2:
        return 1.0

    if pos1[:2] == pos2[:2]:
        return 0.9

    return 0.6


def get_complementarity(player1, player2):

    cols = [
        "Vision",
        "Short Passing",
        "Long Passing",
        "Finishing",
        "Positioning",
        "Defending",
        "Interceptions"
    ]

    cols = [
        c for c in cols
        if c in df.columns
    ]

    p1 = pd.to_numeric(
        player1[cols],
        errors="coerce"
    ).fillna(0)

    p2 = pd.to_numeric(
        player2[cols],
        errors="coerce"
    ).fillna(0)

    diff = abs(
        p1.mean()
        -
        p2.mean()
    )

    score = 1 - diff / 100

    return max(
        0,
        min(score, 1)
    )


def chemistry_score(idx1, idx2):

    similarity = similarity_matrix[idx1][idx2]

    position_fit = get_position_fit(
        df.iloc[idx1]["Pos"],
        df.iloc[idx2]["Pos"]
    )

    complementarity = get_complementarity(
        df.iloc[idx1],
        df.iloc[idx2]
    )

    return (
        0.5 * similarity
        +
        0.3 * position_fit
        +
        0.2 * complementarity
    )


def get_similar_players(
    player_idx,
    top_n=5
):

    scores = []

    for idx in range(len(df)):

        if idx == player_idx:
            continue

        scores.append(
            (
                idx,
                similarity_matrix[player_idx][idx]
            )
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:top_n]


def get_compatible_partners(
    player_idx,
    top_n=5
):

    scores = []

    for idx in range(len(df)):

        if idx == player_idx:
            continue

        scores.append(
            (
                idx,
                chemistry_score(
                    player_idx,
                    idx
                )
            )
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:top_n]


def get_archetype(player_idx):

    cluster = archetypes.iloc[
        player_idx
    ]["Cluster"]

    return ARCHETYPE_NAMES.get(
        cluster,
        f"Cluster {cluster}"
    )


def recommend_unit(player_idx):

    partners = get_compatible_partners(
        player_idx,
        top_n=2
    )

    unit = [
        df.iloc[player_idx]["Player"]
    ]

    for idx, _ in partners:
        unit.append(
            df.iloc[idx]["Player"]
        )

    return unit


def football_report(player_name):

    player_idx = get_player_index(
        player_name
    )

    if player_idx is None:
        print(
            "Player not found."
        )
        return

    player = df.iloc[player_idx]

    print("\n")
    print("=" * 50)
    print("FOOTBALL INTELLIGENCE REPORT")
    print("=" * 50)

    print(
        f"\nPlayer: {player['Player']}"
    )

    print(
        f"Position: {player['Pos']}"
    )

    print(
        f"Archetype: {get_archetype(player_idx)}"
    )

    print(
        "\nTop Similar Players"
    )

    similar = get_similar_players(
        player_idx
    )

    for rank, (idx, score) in enumerate(
        similar,
        start=1
    ):
        print(
            f"{rank}. "
            f"{df.iloc[idx]['Player']} "
            f"({score:.3f})"
        )

    print(
        "\nTop Compatible Partners"
    )

    partners = get_compatible_partners(
        player_idx
    )

    for rank, (idx, score) in enumerate(
        partners,
        start=1
    ):
        print(
            f"{rank}. "
            f"{df.iloc[idx]['Player']} "
            f"({score:.3f})"
        )

    print(
        "\nRecommended Unit"
    )

    unit = recommend_unit(
        player_idx
    )

    for player_name in unit:
        print(
            f"- {player_name}"
        )
    cluster = archetypes.iloc[player_idx]["Cluster"]
    print(f"Cluster: {cluster}")
    print(f"Archetype: {get_archetype(player_idx)}")


player_input = input(
    "Enter player name: "
)

football_report(
    player_input
)