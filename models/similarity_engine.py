import sys
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("data/final_merged_dataset.csv")

df = df.drop_duplicates(subset=["Player"])
df = df.reset_index(drop=True)

embeddings = np.load("models/latent_embeddings.npy")

similarity_matrix = cosine_similarity(embeddings)

player_name = "Michael Olise"

matching_players = df[
    df["Player"]
    .str.lower()
    .str.contains(player_name.lower(), na=False)
]

if matching_players.empty:

    print("Player not found!")

    exit()

player_index = matching_players.index[0]

print("\nMatched Player:")
print(df.iloc[player_index]["Player"])

player_position = df.iloc[player_index]["Pos"]

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

top_players = similar_scores[:5]

print(f"\nTop players similar to {player_name}:\n")

for idx, score in top_players:

    print(
        f"{df.iloc[idx]['Player']} -> Similarity: {score:.3f}"
    )