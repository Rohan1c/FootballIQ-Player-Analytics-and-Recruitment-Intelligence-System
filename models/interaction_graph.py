import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

df = pd.read_csv("data/final_merged_dataset.csv")

embeddings = np.load("models/latent_embeddings.npy")

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


def build_interaction_graph(
    player_name,
    top_n=5
):

    player_index = get_player_index(
        player_name
    )

    if player_index is None:
        print("Player not found")
        return

    player = df.iloc[player_index]["Player"]

    scores = []

    for idx in range(len(df)):

        if idx == player_index:
            continue

        score = similarity_matrix[
            player_index
        ][idx]

        scores.append(
            (
                idx,
                score
            )
        )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_players = scores[:top_n]

    G = nx.Graph()

    G.add_node(player)

    for idx, score in top_players:

        other_player = df.iloc[idx]["Player"]

        G.add_node(other_player)

        G.add_edge(
            player,
            other_player,
            weight=round(score, 3)
        )

    pos = nx.spring_layout(
        G,
        seed=42
    )

    edge_labels = nx.get_edge_attributes(
        G,
        "weight"
    )

    plt.figure(
        figsize=(10, 8)
    )

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        font_size=8
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels
    )

    plt.title(
        f"Interaction Graph: {player}"
    )

    plt.show()


player_name = input(
    "Enter player name: "
)

build_interaction_graph(
    player_name
)