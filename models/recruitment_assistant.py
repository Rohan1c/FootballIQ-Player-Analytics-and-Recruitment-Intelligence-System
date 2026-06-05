import sys
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances

sys.stdout.reconfigure(encoding="utf-8")

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


def chemistry_score(idx1, idx2):

    similarity = similarity_matrix[idx1][idx2]

    pos1 = str(df.iloc[idx1]["Pos"])
    pos2 = str(df.iloc[idx2]["Pos"])

    if pos1 == pos2:
        position_fit = 1.0
    else:
        position_fit = 0.6

    return (
        0.7 * similarity
        +
        0.3 * position_fit
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


def get_replacements(
    player_idx,
    top_n=5
):

    target_pos = str(
        df.iloc[player_idx]["Pos"]
    )

    scores = []

    for idx in range(len(df)):

        if idx == player_idx:
            continue

        player_pos = str(
            df.iloc[idx]["Pos"]
        )

        if player_pos != target_pos:
            continue

        similarity = similarity_matrix[
            player_idx
        ][idx]

        scores.append(
            (
                idx,
                similarity
            )
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:top_n]


def get_young_replacements(
    player_idx,
    age_limit=25,
    top_n=5
):

    target_pos = str(
        df.iloc[player_idx]["Pos"]
    )

    scores = []

    for idx in range(len(df)):

        if idx == player_idx:
            continue

        player_pos = str(
            df.iloc[idx]["Pos"]
        )

        if player_pos != target_pos:
            continue

        age = pd.to_numeric(
            df.iloc[idx]["Age_y"],
            errors="coerce"
        )

        if pd.isna(age):
            continue

        if age > age_limit:
            continue

        similarity = similarity_matrix[
            player_idx
        ][idx]

        scores.append(
            (
                idx,
                similarity
            )
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:top_n]


def get_partners(
    player_idx,
    top_n=5
):

    scores = []

    for idx in range(len(df)):

        if idx == player_idx:
            continue

        score = chemistry_score(
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


def print_section(
    title,
    players
):

    print(f"\n{title}")
    print("-" * len(title))

    for rank, (idx, score) in enumerate(
        players,
        start=1
    ):

        print(
            f"{rank}. "
            f"{df.iloc[idx]['Player']} "
            f"({score:.3f})"
        )


player_name = input(
    "Enter player name: "
)

player_idx = get_player_index(
    player_name
)

if player_idx is None:

    print("Player not found.")
    exit()

player = df.iloc[player_idx]

print("\n")
print("=" * 60)
print("FOOTBALL RECRUITMENT ASSISTANT")
print("=" * 60)

print(
    f"\nPlayer: {player['Player']}"
)

print(
    f"Position: {player['Pos']}"
)

print_section(
    "Top Similar Players",
    get_similar_players(
        player_idx
    )
)

print_section(
    "Top Replacements",
    get_replacements(
        player_idx
    )
)

print_section(
    "Top Compatible Partners",
    get_partners(
        player_idx
    )
)

print_section(
    "Young Talents (<25)",
    get_young_replacements(
        player_idx
    )
)