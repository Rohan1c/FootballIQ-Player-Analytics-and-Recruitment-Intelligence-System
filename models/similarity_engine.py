import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("data/players_data_light.csv")

embeddings = np.load("models/latent_embeddings.npy")

similarity_matrix = cosine_similarity(embeddings)

player_name = "Erling Haaland"

player_index = df[df["Player"] == player_name].index[0]

player_position = df.iloc[player_index]["Pos"]

same_position_indices = df[
    df["Pos"].str.contains(
        player_position.split(",")[0],
        na=False
    )
].index

similar_scores = []

for idx in same_position_indices:

    similarity = similarity_matrix[player_index][idx]

    similar_scores.append((idx, similarity))

similar_scores = sorted(
    similar_scores,
    key=lambda x: x[1],
    reverse=True
)

print(f"\nTop players similar to {player_name}:\n")

count = 0

for idx, score in similar_scores:

    similar_player = df.iloc[idx]["Player"]

    if similar_player != player_name:

        print(
            similar_player,
            "-> Similarity:",
            round(score, 3)
        )

        count += 1

    if count == 5:
        break