import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances

df = pd.read_csv(
    "data/final_merged_dataset.csv"
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

    diff = np.abs(
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


def build_squad(player_name, squad_size=11):

    player_idx = get_player_index(
        player_name
    )

    if player_idx is None:
        print("Player not found.")
        return

    anchor_player = df.iloc[
        player_idx
    ]["Player"]

    scores = []

    for idx in range(len(df)):

        if idx == player_idx:
            continue

        chemistry = chemistry_score(
            player_idx,
            idx
        )

        scores.append(
            (
                idx,
                chemistry
            )
        )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    squad = scores[: squad_size - 1]

    print(
        f"\nRecommended Squad Around {anchor_player}\n"
    )

    print(
        f"{anchor_player} (Anchor Player)"
    )

    total_chemistry = 0

    for idx, chemistry in squad:

        player = df.iloc[idx]["Player"]

        position = df.iloc[idx]["Pos"]

        print(
            f"{player} | {position} | Chemistry: {chemistry:.3f}"
        )

        total_chemistry += chemistry

    avg_chemistry = (
        total_chemistry /
        len(squad)
    )

    print(
        f"\nAverage Squad Chemistry: {avg_chemistry:.3f}"
    )


player_name = input(
    "Enter anchor player: "
)

build_squad(
    player_name
)