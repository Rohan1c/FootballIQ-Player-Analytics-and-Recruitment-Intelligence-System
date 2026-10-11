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

import numpy as np
import matplotlib.pyplot as plt

from legacy.models.similarity_engine import (
    get_similar_players
)

player_name = input(
    "Enter player name: "
)

results = get_similar_players(
    player_name,
    top_n=10
)

if not results:

    print(
        "No similar players found."
    )

    exit()

names = []
scores = []

for player, score in results:

    names.append(player)

    scores.append(score)

plt.style.use("ggplot")

fig, ax = plt.subplots(
    figsize=(14,8)
)

colors = plt.cm.viridis(

    np.linspace(

        0.25,

        0.95,

        len(scores)

    )

)

bars = ax.barh(

    names,

    scores,

    color=colors,

    edgecolor="black",

    linewidth=1.2

)

ax.invert_yaxis()

for bar in bars:

    width = bar.get_width()

    ax.text(

        width + 0.002,

        bar.get_y()
        + bar.get_height()/2,

        f"{width:.3f}",

        va="center",

        fontsize=10,

        fontweight="bold"

    )

best_player = names[0]
best_score = scores[0]

ax.set_title(

    f"FootballIQ Similarity Analysis\nTop Similar Players to {player_name}",

    fontsize=20,

    fontweight="bold",

    pad=20

)

ax.set_xlabel(

    "Similarity Score",

    fontsize=13,

    fontweight="bold"

)

ax.set_ylabel(

    "Players",

    fontsize=13,

    fontweight="bold"

)

ax.grid(

    axis="x",

    linestyle="--",

    alpha=0.4

)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.figtext(

    0.5,

    0.01,

    f"Closest Match: {best_player} ({best_score:.3f})",

    ha="center",

    fontsize=11,

    fontweight="bold"

)

plt.tight_layout()

plt.show()