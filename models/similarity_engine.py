import sys
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("data/final_merged_dataset.csv")

df = df.drop_duplicates(subset=["Player"])

df = df.reset_index(drop=True)

embeddings = np.load("models/latent_embeddings.npy")


def apply_position_weights(player_position, embeddings):

    weighted_embeddings = embeddings.copy()

    if "FW" in player_position:

        weighted_embeddings *= 1.15

    elif "MF" in player_position:

        weighted_embeddings *= 1.10

    elif "DF" in player_position:

        weighted_embeddings *= 1.20

    return weighted_embeddings


def get_similar_players(player_name, top_n=5):

    exact_matches = df[
        df["Player"]
        .str.lower()
        ==
        player_name.lower()
    ]

    if not exact_matches.empty:

        matching_players = exact_matches

    else:

        matching_players = df[
            df["Player"]
            .str.lower()
            .str.contains(player_name.lower(), na=False)
        ]

    if matching_players.empty:

        print("Player not found!")

        return

    player_index = matching_players.index[0]

    matched_name = df.iloc[player_index]["Player"]

    print(f"\nMatched Player: {matched_name}")

    player_position = df.iloc[player_index]["Pos"]

    weighted_embeddings = apply_position_weights(
        player_position,
        embeddings
    )

    distance_matrix = euclidean_distances(
    weighted_embeddings
    )

    max_dist = np.max(distance_matrix)

    similarity_matrix = (
        1 - distance_matrix / max_dist
    )

    same_position_indices = df[
        df["Pos"].str.contains(
            player_position.split(",")[0],
            na=False
        )
    ].index

    similar_scores = []

    for idx in same_position_indices:

        if idx != player_index:

            score = similarity_matrix[player_index][idx]

            similar_scores.append((idx, score))

    similar_scores = sorted(
        similar_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_players = similar_scores[:top_n]

    print(f"\nTop players similar to {matched_name}:\n")

    for idx, score in top_players:

        print(
            f"{df.iloc[idx]['Player']} -> Similarity: {score:.3f}"
        )


player_input = input("Enter player name: ")

get_similar_players(player_input)