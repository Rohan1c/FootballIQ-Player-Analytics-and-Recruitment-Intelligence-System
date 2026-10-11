import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import networkx as nx
import matplotlib.pyplot as plt

from legacy.models.compatibility_engine import (
    compatibility_score,
    find_player,
    df
)

from legacy.models.prototype_role_engine import (
    get_primary_role
)

players = [

    "Pedri",
    "Rodri",
    "Valverde",
    "Bellingham",
    "Palmer",
    "Caicedo"

]

G = nx.Graph()

player_roles = {}

for player_name in players:

    idx = find_player(player_name)

    if idx is None:
        continue

    player = df.iloc[idx]

    role = get_primary_role(
        player
    )

    player_roles[player_name] = role

    G.add_node(
        player_name
    )

for i in range(len(players)):

    for j in range(i + 1, len(players)):

        idx1 = find_player(players[i])
        idx2 = find_player(players[j])

        if idx1 is None or idx2 is None:
            continue

        score = compatibility_score(
            idx1,
            idx2
        )

        if score >= 0.67:

            G.add_edge(

                players[i],

                players[j],

                weight=score

            )

role_colors = {

    "Creative Playmaker": "#FFD700",

    "Deep Playmaker": "#00BFFF",

    "Ball Winner": "#FF4D4D",

    "Box-to-Box": "#32CD32",

    "Creative Winger": "#FF69B4",

    "Wide Winger": "#BA55D3",

    "Inside Forward": "#FF8C00",

    "Poacher": "#FFA500",

    "False 9": "#9370DB",

    "Target Forward": "#8B4513",

    "Ball Playing Defender": "#20B2AA",

    "Defensive Defender": "#708090",

    "Goalkeeper": "#000000"

}

node_colors = []

for node in G.nodes():

    role = player_roles.get(
        node,
        ""
    )

    node_colors.append(

        role_colors.get(
            role,
            "#A9A9A9"
        )

    )

node_sizes = []

for node in G.nodes():

    degree = G.degree(node)

    node_sizes.append(

        1800 + degree * 700

    )

pos = nx.kamada_kawai_layout(
    G
)

weights = [

    G[u][v]["weight"] * 10

    for u, v in G.edges()
]

plt.figure(
    figsize=(14, 10)
)

nx.draw_networkx_nodes(

    G,

    pos,

    node_size=node_sizes,

    node_color=node_colors,

    edgecolors="black",

    linewidths=2,

    alpha=0.95

)

nx.draw_networkx_labels(

    G,

    pos,

    font_size=10,

    font_weight="bold"

)

nx.draw_networkx_edges(

    G,

    pos,

    width=weights,

    alpha=0.7

)

edge_labels = {}

for u, v in G.edges():

    score = G[u][v]["weight"]

    edge_labels[(u, v)] = f"{score:.2f}"

nx.draw_networkx_edge_labels(

    G,

    pos,

    edge_labels=edge_labels,

    font_size=8

)

legend_items = []

for role, color in role_colors.items():

    legend_items.append(

        plt.Line2D(

            [0],

            [0],

            marker='o',

            color='w',

            markerfacecolor=color,

            markersize=10,

            label=role

        )

    )

plt.legend(

    handles=legend_items,

    bbox_to_anchor=(1.05, 1),

    loc="upper left",

    fontsize=8

)

plt.title(

    "FootballIQ Squad Compatibility Network",

    fontsize=20,

    fontweight="bold"

)

plt.figtext(

    0.5,

    0.02,

    "Node Size = Connectivity | Edge Weight = Compatibility Score | Node Color = FootballIQ Role",

    ha="center",

    fontsize=10

)

plt.axis("off")

plt.tight_layout()

plt.show()