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


def get_position_fit(pos1, pos2):

    pos1 = str(pos1)
    pos2 = str(pos2)

    if pos1 == pos2:
        return 1.0

    if pos1[0:2] == pos2[0:2]:
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

    available = [
        c for c in cols
        if c in df.columns
    ]

    p1 = pd.to_numeric(
        player1[available],
        errors="coerce"
    ).fillna(0)

    p2 = pd.to_numeric(
        player2[available],
        errors="coerce"
    ).fillna(0)

    diff = np.abs(
        p1.mean()
        -
        p2.mean()
    )

    score = 1 - (diff / 100)

    return max(
        0,
        min(score, 1)
    )


def chemistry_score(
    idx1,
    idx2
):

    similarity = similarity_matrix[
        idx1
    ][idx2]

    position_fit = get_position_fit(
        df.iloc[idx1]["Pos"],
        df.iloc[idx2]["Pos"]
    )

    complementarity = get_complementarity(
        df.iloc[idx1],
        df.iloc[idx2]
    )

    chemistry = (
        0.5 * similarity
        +
        0.3 * position_fit
        +
        0.2 * complementarity
    )

    return chemistry


def build_chemistry_graph(
    player_name,
    top_n=5
):

    player_index = get_player_index(
        player_name
    )

    if player_index is None:
        print("Player not found")
        return

    player = df.iloc[
        player_index
    ]["Player"]

    scores = []

    for idx in range(len(df)):

        if idx == player_index:
            continue

        score = chemistry_score(
            player_index,
            idx
        )

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

    print(
        f"\nTop Chemistry Partners for {player}\n"
    )

    for idx, score in top_players:

        other = df.iloc[idx]["Player"]

        print(
            f"{other} -> {score:.3f}"
        )

        G.add_edge(
            player,
            other,
            weight=round(score, 3)
        )

    pos = nx.spring_layout(
        G,
        seed=42
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

    edge_labels = nx.get_edge_attributes(
        G,
        "weight"
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels
    )

    plt.title(
        f"Chemistry Graph: {player}"
    )

    plt.show()

    print(
        "\nDegree Centrality"
    )

    centrality = nx.degree_centrality(
        G
    )

    for node, score in sorted(
        centrality.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"{node}: {score:.3f}"
        )


player_name = input(
    "Enter player name: "
)

build_chemistry_graph(
    player_name
)